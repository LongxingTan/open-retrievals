import json
import logging
import os
from abc import ABC, abstractmethod
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


class BaseRanker(ABC, torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super(BaseRanker, self).__init__()
        self.model: Optional[nn.Module] = model
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Pytorch forward method."""
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode documents."""
        pass

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def preprocess(self, batch_sentence_pair, query_max_length, document_max_length):
        query_list = [item[0] for item in batch_sentence_pair]
        document_list = [item[1] for item in batch_sentence_pair]

        query_batch_tokens = self.tokenizer(
            query_list, padding='max_length', truncation=True, max_length=query_max_length, return_tensors='pt'
        )
        query_batch_tokens_on_device = {k: v.to(self.device) for k, v in query_batch_tokens.items()}
        document_batch_tokens = self.tokenizer(
            document_list, padding='max_length', truncation=True, max_length=document_max_length, return_tensors='pt'
        )
        document_batch_tokens_on_device = {k: v.to(self.device) for k, v in document_batch_tokens.items()}

        return {
            "query_input_ids": query_batch_tokens_on_device['input_ids'],
            "query_attention_mask": query_batch_tokens_on_device['attention_mask'],
            "doc_input_ids": document_batch_tokens_on_device['input_ids'],
            "doc_attention_mask": document_batch_tokens_on_device['attention_mask'],
        }


class AutoModelForRanking(BaseRanker):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        train_group_size: int = 1,
        pooling_method: Optional[str] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            assert ValueError("Please use AutoModelForRanking.from_pretrained(model_name_or_path)")

        self.model: Optional[nn.Module] = model
        self.tokenizer = tokenizer
        self.train_group_size = train_group_size
        self.pooling_method = pooling_method
        if pooling_method:
            self.pooling = AutoPooling(self.pooling_method)

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
        self.temperature = temperature

        if device is None:
            self.device = get_device_name()
        else:
            self.device = device

        # self._post_init()
        self.to(self.device)

    # def _post_init(self):
    #     num_features: int = self.model.config.hidden_size
    #     self.classifier = nn.Linear(num_features, 1)
    #     try:
    #         state_dict = torch.load(os.path.join('./', "colbert_linear"), map_location=self.device)
    #         self.dense_pooler.load_state_dict(state_dict)
    #     except FileNotFoundError:
    #         self._init_weights(self.classifier)
    #         logger.warning("Could not find dense weight, initialize it randomly")

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

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Union[torch.Tensor, SequenceClassifierOutput]:
        model_output: SequenceClassifierOutput = self.model(input_ids, attention_mask, output_hidden_states=True)

        if not self.pooling_method:
            return model_output

        if 'last_hidden_state' in model_output:
            last_hidden_state = model_output['last_hidden_state']
        elif 'hidden_states' not in model_output:
            last_hidden_state = model_output[0]
        else:
            raise ValueError

        if self.pooling is not None:
            embeddings = self.pooling(last_hidden_state, attention_mask)
        else:
            embeddings = self.classifier(last_hidden_state)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        if self.temperature is not None:
            features = features / self.temperature

        if not self.training:
            return features

        logits = features.logits
        scores = logits.view(-1, self.train_group_size)

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

            loss = self.loss_fn(scores.squeeze(), labels.squeeze().float())
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
        model_type = model_type.lower().replace('-', '')
        logger.info(f'Set model type: {model_type}')
        model_class = {'crossencoder': self, 'colbert': ColBERT}
        model_class = model_class.get(model_type)
        return model_class(
            model=self.model,
            tokenizer=self.tokenizer,
            pooling_method=self.pooling_method,
            loss_fn=self.loss_fn,
            loss_type=self.loss_type,
            **kwargs,
        )

    @torch.no_grad()
    def compute_score(
        self,
        text_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
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

        scores_list: List[float] = []
        for i in tqdm(range(0, len(text_pairs), batch_size), desc="Scoring", disable=not show_progress_bar):
            if isinstance(text_pairs[0][0], str):
                batch = self.tokenizer(
                    text_pairs[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            else:
                batch = self.tokenizer.pad(
                    text_pairs[i : i + batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors='pt',
                )

            batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
            scores = self.model(**batch_on_device, return_dict=True).logits.view(-1).float()
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
        if query is None or len(query) == 0 or len(documents) == 0:
            return {'rerank_documents': [], 'rerank_scores': []}

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
        pooling_method: Optional[str] = None,
        num_labels: int = 1,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        causal_lm: bool = False,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_lora: bool = False,
        lora_config=None,
        device: Optional[str] = None,
        linear_dim: int = 1,
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, trust_remote_code=trust_remote_code, **kwargs
        )

        if use_fp16:
            model.half()

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            if not lora_config:
                raise ValueError("If use_lora is true, please provide a valid lora_config from peft")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        reranker = cls(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            device=device,
            loss_fn=loss_fn,
            loss_type=loss_type,
            linear_dim=linear_dim,
        )
        return reranker

    def save_pretrained(self, path: str, safe_serialization: bool = True):
        """
        Saves all model and tokenizer to path
        """
        logger.info("Save model to {}".format(path))
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(path)


class ColBERT(BaseRanker):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        linear_layer: Optional[nn.Module] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        max_length: Optional[int] = None,
        temperature: Optional[float] = 0.02,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.linear = linear_layer

        self.loss_fn = loss_fn
        self.loss_type = loss_type
        self.max_length = max_length
        self.temperature = temperature

        if device is None:
            self.device = get_device_name()
        else:
            self.device = device

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.model(input_ids, attention_mask, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            # [batch, seq_len, attention_dim]
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[1]

        embeddings = hidden_state * attention_mask.unsqueeze(-1)
        if self.linear is not None:
            embeddings = self.linear(embeddings)

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
        return embeddings

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
        query_embedding = self.encode(query_input_ids, query_attention_mask, normalize=True)
        positive_embedding = self.encode(pos_input_ids, pos_attention_mask, normalize=True)
        scores = self.score(query_embedding, positive_embedding)

        if self.training:
            scores = scores.unsqueeze(-1)
            if neg_input_ids is not None:
                negative_embedding = self.encode(neg_input_ids, neg_attention_mask, normalize=True)
                negative_scores = self.score(query_embedding, negative_embedding)
                negative_scores = negative_scores.unsqueeze(-1)

                scores = torch.cat([scores, negative_scores], dim=-1)

            # if self.temperature is not None:
            #     scores = scores / self.temperature

            labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
            loss = self.loss_fn(scores, labels)

            if return_dict:
                outputs_dict = dict()
                outputs_dict['loss'] = loss
                return outputs_dict
            return loss

        return scores

    @torch.no_grad()
    def compute_score(
        self,
        text_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 128,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()
        if isinstance(text_pairs[0], str):
            text_pairs = [text_pairs]

        batch_size = min(batch_size, len(text_pairs))
        scores_list: List[float] = []
        for i in tqdm(range(0, len(text_pairs), batch_size), desc="Scoring", disable=not show_progress_bar):
            batch_on_device = self.preprocess(
                text_pairs[i : i + batch_size], query_max_length=max_length, document_max_length=max_length
            )
            query_embedding = self.encode(
                batch_on_device['query_input_ids'], batch_on_device['query_attention_mask'], normalize=True
            )
            doc_embedding = self.encode(
                batch_on_device['doc_input_ids'], batch_on_device['doc_attention_mask'], normalize=True
            )
            scores = self.score(query_embedding, doc_embedding)
            if normalize:
                scores = torch.sigmoid(scores)
            scores_list.extend(scores.cpu().numpy().tolist())

        if len(scores_list) == 1:
            return scores_list[0]
        return scores_list

    def score(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ):
        late_interactions = torch.einsum(
            "bsh,bth->bst",
            query_embeddings,
            document_embeddings,
        )
        late_interactions = late_interactions.max(2).values.sum(1)
        return late_interactions

    def save_pretrained(self, save_directory: Union[str, os.PathLike], safe_serialization: bool = True):
        logger.info("Save model to {}".format(save_directory))
        state_dict_fn = lambda state_dict: type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        state_dict = self.model.state_dict()
        self.model.save_pretrained(
            save_directory, state_dict=state_dict_fn(state_dict), safe_serialization=safe_serialization
        )
        torch.save(state_dict_fn(self.linear.state_dict()), os.path.join(save_directory, 'colbert_linear.pt'))
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        causal_lm: bool = False,
        trust_remote_code: bool = True,
        colbert_dim: int = 768,
        loss_fn: Union[nn.Module, Callable] = nn.CrossEntropyLoss(reduction='mean'),
        loss_type: Literal['classification', 'regression'] = 'classification',
        device: Optional[str] = None,
        **kwargs,
    ):
        if not model_name_or_path or not isinstance(model_name_or_path, str):
            assert ValueError('Please input valid model_name_or_path')

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
        # model = AutoModelForSequenceClassification(
        #     model_name_or_path, num_labels=colbert_dim, trust_remote_code=trust_remote_code
        # )
        linear_layer = nn.Linear(model.config.hidden_size, colbert_dim, dtype=torch.float32, bias=False)

        if os.path.exists(path=os.path.join(model_name_or_path, 'colbert_linear.pt')):
            logger.info(f'Loading colbert_linear weight from {model_name_or_path}')
            colbert_state_dict = torch.load(os.path.join(model_name_or_path, 'colbert_linear.pt'), map_location='cpu')
            linear_layer.load_state_dict(colbert_state_dict)
        else:
            torch.nn.init.xavier_uniform_(tensor=linear_layer.weight)

        if os.path.exists(path=os.path.join(model_name_or_path, "metadata.json")):
            with open(file=os.path.join(model_name_or_path, "metadata.json"), mode="r") as f:
                metadata = json.load(fp=f)
            max_length = metadata["max_length"]
            print(max_length)

        ranker = cls(
            model=model,
            tokenizer=tokenizer,
            linear_layer=linear_layer,
            device=device,
            loss_fn=loss_fn,
            loss_type=loss_type,
        )
        return ranker


class LLMRanker(BaseRanker):
    def __init__(self, model, tokenizer):
        super(LLMRanker, self).__init__()


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

    def _merge_inputs(self, chunk1_raw, chunk2, sep_id: int):
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
