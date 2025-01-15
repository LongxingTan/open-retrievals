"""Base model for embedding and reranking"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm.auto import tqdm, trange
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class Base(ABC, torch.nn.Module):
    """Base class for embedding and reranking model"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            assert ValueError("Please use AutoModelForEmbedding.from_pretrained(model_name_or_path)")
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Pytorch forward method."""

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

    def _sort_by_length(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Sorts sentence pairs by total length."""
        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        return [sentence_pairs[idx] for idx in length_sorted_idx]


class BaseRanker(Base):
    def __init__(self, model: Optional[nn.Module] = None, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(model, tokenizer)

    def score(self):
        """score for ranking"""

    def preprocess_pair(
        self,
        batch_sentence_pair: List[List[str, str]],
        query_max_length: int,
        document_max_length: int,
        padding='max_length',
    ):
        """Preprocesses the pair of sentences (query, document) for the model."""
        query_list = [item[0] for item in batch_sentence_pair]
        document_list = [item[1] for item in batch_sentence_pair]

        query_batch_tokens = self.tokenizer(
            query_list, padding=padding, truncation=True, max_length=query_max_length, return_tensors='pt'
        )
        document_batch_tokens = self.tokenizer(
            document_list, padding=padding, truncation=True, max_length=document_max_length, return_tensors='pt'
        )

        return {
            "query_input_ids": query_batch_tokens['input_ids'].to(self.device),
            "query_attention_mask": query_batch_tokens['attention_mask'].to(self.device),
            "doc_input_ids": document_batch_tokens['input_ids'].to(self.device),
            "doc_attention_mask": document_batch_tokens['attention_mask'].to(self.device),
        }

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ) -> Union[List[float], float]:
        """compute scores for a list of sentence pairs."""
        self.model.eval()
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        sentences_sorted = self._sort_by_length(sentence_pairs)

        if self.query_instruction or self.document_instruction:
            sentences_sorted = [
                (self.query_instruction + pair[0], self.document_instruction + pair[1]) for pair in sentences_sorted
            ]

        all_scores: List[float] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]
            batch_on_device = self.preprocess_pair(
                batch_sentences, query_max_length=max_length, document_max_length=max_length
            )

            scores = self._encode_and_score(batch_on_device)

            if normalize:
                scores = torch.sigmoid(scores)

            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [
            all_scores[idx]
            for idx in np.argsort(np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs]))
        ]

        return all_scores[0] if len(all_scores) == 1 else all_scores
