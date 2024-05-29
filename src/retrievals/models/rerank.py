import logging
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)

from ..data.collator import RerankCollator
from .pooling import AutoPooling
from .utils import check_casual_lm, get_device_name

logger = logging.getLogger(__name__)


class RerankModel(nn.Module):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        pooling_method: str = 'mean',
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.model: Optional[nn.Module] = model
        self.tokenizer = tokenizer
        self.pooling = AutoPooling(pooling_method)

        if self.model:
            num_features: int = self.model.config.hidden_size
            self.classifier = nn.Linear(num_features, 1)
            self._init_weights(self.classifier)
        self.loss_fn = loss_fn
        self.loss_type = loss_type

        if max_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_length = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_length = max_length

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.model(input_ids, attention_mask, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[1]

        if self.pooling:
            embeddings = self.pooling(hidden_state, attention_mask)
        else:
            embeddings = self.classifier(hidden_state[:, 1:])
        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if input_ids is not None:
            features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        elif inputs is not None:
            features = self.encode(**inputs)
        else:
            raise ValueError("input_ids(tensor) and inputs(dict) can't be empty as the same time")
        logits = self.classifier(features).reshape(-1)

        if return_dict:
            outputs_dict = dict()
            outputs_dict['logits'] = logits

        if labels is not None:
            if not self.loss_fn:
                if self.loss_type == 'regression':
                    logits = torch.sigmoid(logits)
                    self.loss_fn = nn.MSELoss()

                elif self.loss_type == 'classification':
                    self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            loss = self.loss_fn(logits, labels.float())
            if return_dict:
                outputs_dict['loss'] = loss
                return outputs_dict
            else:
                return logits, loss
        else:
            if return_dict:
                return outputs_dict
            return logits

    def set_model_type(self, model_type: Literal['cross-encoder', 'colbert'], **kwargs):
        model_class = {'cross-encoder': self, 'colbert': ColBERT}
        model_class = model_class.get(model_type.lower())
        return model_class(**kwargs)

    @torch.no_grad()
    def compute_score(
        self,
        text_pairs: Union[List[Tuple[str, str]], Tuple[str, str], List[str]],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 128,
        max_length: int = 512,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()
        if isinstance(text_pairs[0], str):
            text_pairs = [text_pairs]

        batch_size = min(batch_size, len(text_pairs))

        if not data_collator:
            data_collator = RerankCollator(tokenizer=self.tokenizer)

        scores_list: List[float] = []
        for i in range(0, len(text_pairs), batch_size):
            text_batch = [{'query': text_pairs[i][0], 'document': text_pairs[i][1]} for i in range(i, i + batch_size)]
            batch = data_collator(text_batch)
            scores = self.model(batch['input_ids'], batch['attention_mask'], return_dict=True).logits.view(-1).float()
            if normalize:
                scores = torch.sigmoid(scores)
            scores_list.extend(scores.cpu().numpy().tolist())

        if len(scores_list) == 1:
            return scores_list[0]
        return scores_list

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 32,
        chunk_max_length: int = 512,
        chunk_overlap: int = 50,
        max_chunks_per_doc: int = 100,
        normalize: bool = False,
        show_progress_bar: bool = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # if isinstance(query, str):
        #     text_pairs = [(query, doc) for doc in document]
        # elif isinstance(query, (list, tuple)):
        #     text_pairs = [(q, doc) for q, doc in zip(query, document)]
        # else:
        #     pass

        splitter = DocumentSplitter(
            chunk_size=chunk_max_length, chunk_overlap=chunk_overlap, max_chunks_per_doc=max_chunks_per_doc
        )
        text_pairs, sentence_pairs_pids = splitter.create_documents(
            query,
            documents,
            tokenizer=self.tokenizer,
        )

        tot_scores = self.compute_score(
            text_pairs=text_pairs,
            data_collator=data_collator,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress_bar,
        )

        merge_scores = [0 for _ in range(len(documents))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_document = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_document.append(documents[mid])

        if return_dict:
            return {
                'rerank_document': sorted_document,
                'rerank_scores': sorted_scores,
                'rerank_ids': merge_scores_argsort.tolist(),
            }
        else:
            return sorted_document

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pooling_method: str = 'mean',
        loss_type: Literal['classification', 'regression'] = 'classification',
        num_labels: int = 1,
        causal_lm: bool = False,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_lora: bool = False,
        lora_config=None,
        device: Optional[str] = None,
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, trust_remote_code=trust_remote_code, **kwargs
        )
        if gradient_checkpointing:
            model.graident_checkpointing_enable()

        if device is None:
            device = get_device_name()
        else:
            device = device

        if use_fp16:
            model.half()
        if use_lora:
            # peft config and wrapping
            from peft import LoraConfig, TaskType, get_peft_model

            if not lora_config:
                raise ValueError("If use_lora is true, please provide a valid lora_config from peft")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        reranker = cls(
            model=model, tokenizer=tokenizer, pooling_method=pooling_method, device=device, loss_type=loss_type
        )
        return reranker

    def save(self, path: str):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path: str):
        """
        Same function to save
        """
        return self.save(path)


class ColBERT(RerankModel):
    def __init__(
        self,
        **kwargs,
    ):
        super(ColBERT, self).__init__(**kwargs)

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor],
        query_attention_mask: Optional[torch.Tensor],
        pos_input_ids: Optional[torch.Tensor],
        pos_attention_mask: Optional[torch.Tensor],
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        query_embedding = self.encode(query_input_ids, query_attention_mask)
        pos_embedding = self.encode(pos_input_ids, pos_attention_mask)

        if neg_input_ids and neg_attention_mask:
            neg_embedding = self.encode(neg_input_ids, neg_attention_mask)
        else:
            neg_embedding = None

        score = self.loss_fn(query_embedding, pos_embedding, neg_embedding)
        if return_dict:
            outputs_dict = dict()
            outputs_dict['score'] = score
            return outputs_dict
        return score


class DocumentSplitter(object):
    """
    Rerank the long document
    - https://github.com/netease-youdao/BCEmbedding/blob/master/BCEmbedding/models/utils.py
    """

    def __init__(self, chunk_size: int, chunk_overlap: int = 0, max_chunks_per_doc: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc

    def create_documents(self, query, documents, tokenizer):
        res_merge_inputs = []
        res_merge_inputs_pids = []

        query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
        sep_id = tokenizer.sep_token_id
        doc_max_length = self.chunk_size - len(query_inputs['input_ids']) - 2

        for pid, document in enumerate(documents):
            document_inputs = tokenizer.encode_plus(document, truncation=False, padding=False, add_special_tokens=False)
            doc_inputs_length = len(document_inputs['input_ids'])

            if doc_inputs_length <= doc_max_length:
                qc_merge_inputs = self._merge_inputs(query_inputs, document_inputs, sep_id)
                res_merge_inputs.append(qc_merge_inputs)
                res_merge_inputs_pids.append(pid)
            else:
                start_id = 0
                while start_id < doc_inputs_length:
                    end_id = start_id + doc_max_length
                    sub_document_inputs = {k: v[start_id:end_id] for k, v in document_inputs.items()}
                    start_id = end_id - self.chunk_overlap if end_id < doc_inputs_length else end_id

                    qp_merge_inputs = self._merge_inputs(query_inputs, sub_document_inputs, sep_id)
                    res_merge_inputs.append(qp_merge_inputs)
                    res_merge_inputs_pids.append(pid)
        return res_merge_inputs, res_merge_inputs_pids

    def _merge_inputs(self, chunk1_raw, chunk2, sep_id):
        chunk1 = deepcopy(chunk1_raw)

        chunk1['input_ids'].append(sep_id)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)

        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])

        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1
