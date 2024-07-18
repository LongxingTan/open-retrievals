import json
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)

from ..data.collator import LLMRerankCollator, RerankCollator
from ..losses.colbert_loss import ColbertLoss
from .base import Base
from .pooling import AutoPooling
from .utils import (
    batch_to_device,
    check_causal_lm,
    find_all_linear_names,
    get_device_name,
)

logger = logging.getLogger(__name__)


class AutoModelForRanking(Base):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        train_group_size: int = 1,
        pooling_method: Optional[str] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        max_length: Optional[int] = None,
        causal_lm: bool = False,
        task_prompt: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
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
        self.causal_lm = causal_lm
        self.task_prompt = task_prompt
        self.query_instruction = query_instruction if query_instruction else ""
        self.document_instruction = document_instruction if document_instruction else ""

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
        # input_ids -> embedding
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
        if self.causal_lm:
            # LLM rerank
            model_output = self.model(input_ids, attention_mask, output_hidden_states=True)
            loss = self.loss_fn(model_output.logits, labels)
            outputs_dict = dict()
            outputs_dict['logits'] = model_output.logits
            outputs_dict['loss'] = loss
            return outputs_dict

        # cross-encoder rerank
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
        model_type = model_type.lower().replace('-', '').replace('_', '')
        logger.info(f'Set model type to: {model_type}')
        model_class = {'crossencoder': self, 'colbert': ColBERT, 'llm': LLMRanker}
        model_class = model_class.get(model_type)
        return model_class(
            model=self.model,
            tokenizer=self.tokenizer,
            loss_fn=self.loss_fn,
            **kwargs,
        )

    def preprocess(
        self,
        batch_sentence_pair: List[List[str]],
        max_length: int,
        padding: Union[str, bool] = 'max_length',
    ):
        if isinstance(batch_sentence_pair[0][0], str):
            batch = self.tokenizer(
                batch_sentence_pair,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            batch = self.tokenizer.pad(
                batch_sentence_pair,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors='pt',
            )
        batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
        return batch_on_device

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        """
        preprocess -> score -> output
        """
        self.model.eval()
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        all_scores: List[float] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]

            batch_on_device = self.preprocess(batch_sentences, max_length=max_length)

            scores = self.model(**batch_on_device, return_dict=True).logits.view(-1).float()

            if normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 16,
        chunk_max_length: int = 256,
        chunk_overlap: int = 48,
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
            sentence_pairs=text_pairs,
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
        lora_path: Optional[str] = None,
        quantization_config=None,
        task_prompt: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        if causal_lm or check_causal_lm(model_name_or_path):
            logger.info('Set model to AutoModelForCausalLM')
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, quantization_config=quantization_config, trust_remote_code=trust_remote_code
            )
        else:
            logger.info('Set model to  AutoModelForSequenceClassification')
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels, trust_remote_code=trust_remote_code, **kwargs
            )

        if use_fp16:
            logger.info('Set model to fp16, please note that if you want fp16 during training, set training_args fp16')
            model.half()

        if use_lora:
            logger.info('Set fine-tuning to LoRA')
            from peft import LoraConfig, TaskType, get_peft_model

            if lora_config is None:
                lora_alpha = 64
                lora_dropout = 0.05
                target_modules = find_all_linear_names(model)
                lora_config = LoraConfig(
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias='none',
                    task_type=TaskType.CAUSAL_LM,
                )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        if lora_path is not None:
            logger.info('Load pretrained with LoRA adapter')
            from peft import LoraConfig, PeftModel

            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()

        reranker = cls(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            device=device,
            loss_fn=loss_fn,
            loss_type=loss_type,
            temperature=temperature,
            causal_lm=causal_lm,
            task_prompt=task_prompt,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
        )
        return reranker


class ColBERT(Base):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        linear_layer: Optional[nn.Module] = None,
        max_length: Optional[int] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.linear = linear_layer

        self.loss_fn = loss_fn if loss_fn else ColbertLoss()
        self.max_length = max_length
        self.device = device or get_device_name()
        self.to(self.device)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        if hasattr(outputs, 'last_hidden_state'):
            # shape: [batch, seq_len, attention_dim]
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
        query_embedding = self.encode(query_input_ids, attention_mask=query_attention_mask, normalize=True)
        positive_embedding = self.encode(pos_input_ids, attention_mask=pos_attention_mask, normalize=True)

        if self.training:
            negative_embedding = None
            if neg_input_ids is not None:
                negative_embedding = self.encode(neg_input_ids, attention_mask=neg_attention_mask, normalize=True)

            loss = self.loss_fn(query_embedding, positive_embedding, negative_embedding)

            if return_dict:
                outputs_dict = dict()
                outputs_dict['loss'] = loss
                return outputs_dict
            return loss

        scores = self.score(query_embedding, positive_embedding)
        return scores

    def preprocess(
        self,
        batch_sentence_pair: List[List[str]],
        query_max_length: int,
        document_max_length: int,
        padding='max_length',
    ):
        query_list = [item[0] for item in batch_sentence_pair]
        document_list = [item[1] for item in batch_sentence_pair]

        query_batch_tokens = self.tokenizer(
            query_list, padding=padding, truncation=True, max_length=query_max_length, return_tensors='pt'
        )
        query_batch_tokens_on_device = {k: v.to(self.device) for k, v in query_batch_tokens.items()}
        document_batch_tokens = self.tokenizer(
            document_list, padding=padding, truncation=True, max_length=document_max_length, return_tensors='pt'
        )
        document_batch_tokens_on_device = {k: v.to(self.device) for k, v in document_batch_tokens.items()}

        return {
            "query_input_ids": query_batch_tokens_on_device['input_ids'],
            "query_attention_mask": query_batch_tokens_on_device['attention_mask'],
            "doc_input_ids": document_batch_tokens_on_device['input_ids'],
            "doc_attention_mask": document_batch_tokens_on_device['attention_mask'],
        }

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        scores_list: List[float] = []
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Scoring", disable=not show_progress_bar):
            batch_on_device = self.preprocess(
                sentence_pairs[i : i + batch_size], query_max_length=max_length, document_max_length=max_length
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
        # pair embedding -> score
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
        torch.save(state_dict_fn(self.linear.state_dict()), os.path.join(save_directory, 'linear.pt'))
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        colbert_dim: int = 768,
        loss_fn: Union[nn.Module, Callable] = ColbertLoss(),
        trust_remote_code: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)

        linear_layer = nn.Linear(model.config.hidden_size, colbert_dim, dtype=torch.float32, bias=False)
        if os.path.exists(path=os.path.join(model_name_or_path, 'linear.pt')):
            logger.info(f'Loading colbert_linear weight from {model_name_or_path}')
            colbert_state_dict = torch.load(os.path.join(model_name_or_path, 'linear.pt'), map_location='cpu')
            linear_layer.load_state_dict(colbert_state_dict)
        else:
            logger.info('Xavier uniform random colbert linear layer')
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
        )
        return ranker


class LLMRanker(AutoModelForRanking):
    def __init__(self, task_prompt: Optional[str] = None, token='Yes', **kwargs):
        super(LLMRanker, self).__init__(**kwargs)
        if task_prompt is None:
            task_prompt = (
                "Given a query A and a passage B, determine whether the passage contains an answer to the query"
                "by providing a prediction of either 'Yes' or 'No'."
            )
        self.task_prompt = task_prompt
        self.prompt_inputs = self.tokenizer(self.task_prompt, return_tensors=None, add_special_tokens=False)[
            'input_ids'
        ]
        sep = "\n"
        self.sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
        self.token_loc = self.tokenizer(token, add_special_tokens=False)['input_ids'][0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        model_output = self.model(input_ids, attention_mask, output_hidden_states=True)
        loss = self.loss_fn(model_output.logits, labels)
        outputs_dict = dict()
        outputs_dict['logits'] = model_output.logits
        outputs_dict['loss'] = loss
        return outputs_dict

    def preprocess(self, batch_sentence_pair: List[List[str]], max_length: int, **kwargs):
        collator = LLMRerankCollator(tokenizer=self.tokenizer, prompt=self.task_prompt, max_length=max_length)
        batch_inputs = collator(batch_sentence_pair)

        batch_inputs = {
            'input_ids': batch_inputs['input_ids'].to(self.device),
            'attention_mask': batch_inputs['attention_mask'].to(self.device),
        }
        return batch_inputs

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str]], Tuple[str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        if self.query_instruction or self.document_instruction:
            sentences_sorted = [
                (self.query_instruction + pair[0], self.document_instruction + pair[1]) for pair in sentences_sorted
            ]

        all_scores: List[float] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]
            batch_on_device = self.preprocess(batch_sentences, max_length=max_length)
            outputs = self.model(**batch_on_device, output_hidden_states=True)
            scores = self.score(outputs.logits, batch_on_device['attention_mask'])

            if normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    def score(self, logits, attention_mask: torch.Tensor):
        scores = AutoPooling('last')(logits, attention_mask)
        scores = scores[:, self.token_loc]
        return scores


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
