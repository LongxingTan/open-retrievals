"""Base model for embedding and reranking"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from ..data.collator import LLMRerankCollator, RerankCollator
from .utils import DocumentSplitter, find_all_linear_names, get_device_name

logger = logging.getLogger(__name__)


class Base(ABC, nn.Module):
    """Base class for embedding and reranking model"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            assert ValueError("Model should be like: AutoModelForEmbedding.from_pretrained(model_name_or_path)")
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_device_name()
        self.to(self.device)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Pytorch forward method."""

    @staticmethod
    def setup_lora(model, lora_config, use_qlora: bool = False):
        """Setup LoRA for the model."""
        from peft import get_peft_model, prepare_model_for_kbit_training

        if not lora_config:
            lora_config = Base._create_default_lora_config(model, lora_r=16, lora_alpha=64, lora_dropout=0.05)

        if use_qlora:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def load_lora_weights(model, lora_path: str):
        """Load pre-trained LoRA weights."""
        from peft import PeftModel

        logger.info(f'Loading LoRA adapter from {lora_path}')
        model = PeftModel.from_pretrained(model, lora_path)
        return model.merge_and_unload()

    @staticmethod
    def _create_default_lora_config(model, lora_r: int, lora_alpha: int, lora_dropout: float):
        """Create default LoRA configuration."""
        from peft import LoraConfig

        target_modules = find_all_linear_names(model)
        logger.info(f'Setting LoRA target modules to {target_modules}, r={lora_r}, alpha={lora_alpha}')
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )

    def save_pretrained(self, path: str, safe_serialization: bool = True):
        """Saves all model and tokenizer to path"""
        logger.info("Save model to {}".format(path))
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def resize_token_embeddings(self, new_num_tokens: Optional = None, pad_to_multiple_of: Optional = None):
        # add new, random embeddings for the new tokens
        self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def push_to_hub(self, hub_model_id: str, private: bool = True, **kwargs):
        """push model to hub

        :param hub_model_id: str, hub model id.
        :param private: bool, whether push to private repo. Default True.
        :param kwargs: other kwargs for `push_to_hub` method.
        """
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.backbone.push_to_hub(hub_model_id, private=private, **kwargs)

    def _dist_gather_tensor(self, tensor: Optional[torch.Tensor]):
        """negatives cross device"""
        if tensor is None:
            return None
        tensor = tensor.contiguous()

        all_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, tensor)

        all_tensors[self.process_rank] = tensor
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def _determine_max_length(self) -> int:
        """Determine the maximum sequence length based on model and tokenizer configurations."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_position_embeddings"):
            return min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        return self.tokenizer.model_max_length

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BaseRanker(Base):
    def __init__(self, model: Optional[nn.Module] = None, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(model, tokenizer)

    def preprocess_pair(
        self,
        batch_sentence_pair: List[Tuple[str, str]],
        query_max_length: int,
        document_max_length: int,
        padding='max_length',
        **kwargs,
    ):
        """Preprocesses the pair of sentences (query, document) for the model."""
        query_list = [pair[0] for pair in batch_sentence_pair]
        document_list = [pair[1] for pair in batch_sentence_pair]
        queries_inputs_batch = self.tokenizer(
            query_list,
            return_tensors=None,
            add_special_tokens=False,
            max_length=query_max_length,
            truncation=True,
            **kwargs,
        )['input_ids']
        passages_inputs_batch = self.tokenizer(
            document_list,
            return_tensors=None,
            add_special_tokens=False,
            max_length=document_max_length,
            truncation=True,
            **kwargs,
        )['input_ids']

        inputs_batch = []
        for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
            item = self.tokenizer.prepare_for_model(
                q_inp,
                d_inp,
                truncation='only_second',
                max_length=document_max_length,
                padding=False,
            )
            inputs_batch.append(item)

        return self.tokenizer.pad(inputs_batch, padding=True, return_tensors='pt', **kwargs).to(self.device)

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ) -> Union[List[float], float]:
        """compute scores for a list of sentence pairs.[(q1, d1), (q2, d2), ...]"""
        self.model.eval()

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        if self.query_instruction or self.document_instruction:
            sentences_sorted = [
                (
                    self.query_instruction.format(pair[0]) if self.query_instruction else pair[0],
                    self.document_instruction.format(pair[1]) if self.document_instruction else pair[1],
                )
                for pair in sentences_sorted
            ]

        all_scores: List[float] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]
            batch_on_device = self.preprocess_pair(
                batch_sentences, query_max_length=max_length, document_max_length=max_length
            )

            scores = self.forward(**batch_on_device).flatten().float()

            if normalize:
                scores = torch.sigmoid(scores)

            all_scores.extend(scores.cpu().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        return all_scores[0] if len(all_scores) == 1 else all_scores

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = None,
        return_dict: bool = True,
        normalize: bool = False,
        data_collator: Optional[RerankCollator] = None,
        long_documents_split: bool = False,
        chunk_max_length: int = 256,
        chunk_overlap: int = 48,
        max_chunks_per_doc: int = 100,
        **kwargs,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """
        query: single string text
        documents: list of string
        """
        if query is None or len(query) == 0 or len(documents) == 0:
            return {'rerank_documents': [], 'rerank_scores': []}

        if long_documents_split:
            splitter = DocumentSplitter(
                chunk_size=chunk_max_length, chunk_overlap=chunk_overlap, max_chunks_per_doc=max_chunks_per_doc
            )
            text_pairs, sentence_pairs_pids = splitter.create_documents(
                query,
                documents,
                tokenizer=self.tokenizer,
            )
        else:
            text_pairs = [[query, doc] for doc in documents]
            sentence_pairs_pids = list(range(len(documents)))

        scores = self.compute_score(
            sentence_pairs=text_pairs,
            data_collator=data_collator,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress_bar,
        )

        merge_scores = [float('-inf') for _ in range(len(documents))]
        for pid, score in zip(sentence_pairs_pids, scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_documents = [documents[i] for i in merge_scores_argsort]
        sorted_scores = [merge_scores[i] for i in merge_scores_argsort]

        if return_dict:
            return {
                'rerank_document': sorted_documents,
                'rerank_scores': sorted_scores,
                'rerank_ids': merge_scores_argsort.tolist(),
            }
        else:
            return sorted_documents
