"""Base model for embedding and reranking.

This module provides base classes for embedding and reranking models,
with common functionality for model loading, saving, and inference.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..data.collator import LLMRerankCollator, RerankCollator
from .utils import DocumentSplitter, find_all_linear_names, get_device_name

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model initialization and training.

    Args:
        query_instruction: Template for query text
        document_instruction: Template for document text
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        normalize: Whether to normalize scores
        show_progress_bar: Whether to show progress bar
    """

    query_instruction: Optional[str] = None
    document_instruction: Optional[str] = None
    max_length: int = 256
    batch_size: int = 16
    normalize: bool = False
    show_progress_bar: Optional[bool] = None


class Base(ABC, nn.Module):
    """Base class for embedding and reranking models.

    This class provides common functionality for model loading, saving,
    and inference, as well as LoRA support.

    Args:
        model: Pre-trained model
        tokenizer: Pre-trained tokenizer
        config: Model configuration
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            raise ValueError(
                "Model should be initialized using from_pretrained method, "
                "e.g., AutoModelForEmbedding.from_pretrained(model_name_or_path)"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ModelConfig(**kwargs)
        self.device = get_device_name()
        self.to(self.device)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            torch.Tensor: Model output
        """
        pass

    @staticmethod
    def setup_lora(
        model: nn.Module,
        lora_config: Optional[Dict[str, Any]] = None,
        use_qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ) -> nn.Module:
        """Setup LoRA for the model.

        Args:
            model: Model to apply LoRA to
            lora_config: LoRA configuration
            use_qlora: Whether to use QLoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout rate

        Returns:
            nn.Module: Model with LoRA applied
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        if not lora_config:
            lora_config = Base._create_default_lora_config(
                model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )

        if use_qlora:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def load_lora_weights(model: nn.Module, lora_path: str) -> nn.Module:
        """Load pre-trained LoRA weights.

        Args:
            model: Model to load weights into
            lora_path: Path to LoRA weights

        Returns:
            nn.Module: Model with loaded weights
        """
        from peft import PeftModel

        logger.info(f'Loading LoRA adapter from {lora_path}')
        model = PeftModel.from_pretrained(model, lora_path)
        return model.merge_and_unload()

    @staticmethod
    def _create_default_lora_config(
        model: nn.Module,
        lora_r: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ) -> Any:
        """Create default LoRA configuration.

        Args:
            model: Model to create config for
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout rate

        Returns:
            Any: LoRA configuration
        """
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

    def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
        """Save model and tokenizer to path.

        Args:
            path: Path to save to
            safe_serialization: Whether to use safe serialization
        """
        logger.info(f"Saving model to {path}")
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Enable gradient checkpointing.

        Args:
            gradient_checkpointing_kwargs: Additional arguments for gradient checkpointing
        """
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def enable_input_require_grads(self, **kwargs) -> None:
        """Enable input gradients."""
        self.model.enable_input_require_grads(**kwargs)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        """Resize token embeddings.

        Args:
            new_num_tokens: New number of tokens
            pad_to_multiple_of: Pad to multiple of this number
        """
        self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def push_to_hub(
        self,
        hub_model_id: str,
        private: bool = True,
        **kwargs,
    ) -> None:
        """Push model to hub.

        Args:
            hub_model_id: Hub model ID
            private: Whether to push to private repo
            **kwargs: Additional arguments for push_to_hub
        """
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.model.push_to_hub(hub_model_id, private=private, **kwargs)

    def _dist_gather_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Gather tensors across devices.

        Args:
            tensor: Tensor to gather

        Returns:
            Optional[torch.Tensor]: Gathered tensor
        """
        if tensor is None:
            return None
        tensor = tensor.contiguous()

        all_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, tensor)

        all_tensors[self.process_rank] = tensor
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def _text_length(self, text: Union[List[int], List[List[int]], Dict[str, List[int]]]) -> int:
        """Get the length of input text.

        Args:
            text: Input text

        Returns:
            int: Text length
        """
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum(len(t) for t in text)

    def _determine_max_length(self) -> int:
        """Determine maximum sequence length.

        Returns:
            int: Maximum sequence length
        """
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_position_embeddings"):
            return min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        return self.tokenizer.model_max_length

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: Module to initialize
        """
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
    """Base class for ranking models.

    This class provides common functionality for ranking models,
    including preprocessing and scoring.

    Args:
        model: Pre-trained model
        tokenizer: Pre-trained tokenizer
        config: Model configuration
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, config, **kwargs)

    def preprocess_pair(
        self,
        batch_sentence_pair: List[Tuple[str, str]],
        query_max_length: int,
        document_max_length: int,
        padding: str = 'max_length',
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess sentence pairs for the model.

        Args:
            batch_sentence_pair: List of (query, document) pairs
            query_max_length: Maximum query length
            document_max_length: Maximum document length
            padding: Padding strategy
            **kwargs: Additional arguments

        Returns:
            Dict[str, torch.Tensor]: Preprocessed inputs
        """
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
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
        **kwargs,
    ) -> Union[List[float], float]:
        """Compute scores for sentence pairs.

        Args:
            sentence_pairs: List of (query, document) pairs
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            normalize: Whether to normalize scores
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments

        Returns:
            Union[List[float], float]: Computed scores
        """
        self.model.eval()

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        # Use config values if not specified
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length
        normalize = normalize if normalize is not None else self.config.normalize
        show_progress_bar = show_progress_bar if show_progress_bar is not None else self.config.show_progress_bar

        # Sort by length for efficiency
        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        # Apply instructions if configured
        if self.config.query_instruction or self.config.document_instruction:
            sentences_sorted = [
                (
                    self.config.query_instruction.format(pair[0]) if self.config.query_instruction else pair[0],
                    self.config.document_instruction.format(pair[1]) if self.config.document_instruction else pair[1],
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

        # Restore original order
        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        return all_scores[0] if len(all_scores) == 1 else all_scores

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
        return_dict: bool = True,
        normalize: Optional[bool] = None,
        data_collator: Optional[RerankCollator] = None,
        long_documents_split: bool = False,
        chunk_max_length: int = 256,
        chunk_overlap: int = 48,
        max_chunks_per_doc: int = 100,
        **kwargs,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Rerank documents for a query.

        Args:
            query: Query text
            documents: List of document texts
            batch_size: Batch size for inference
            show_progress_bar: Whether to show progress bar
            return_dict: Whether to return results as dictionary
            normalize: Whether to normalize scores
            data_collator: Data collator for batching
            long_documents_split: Whether to split long documents
            chunk_max_length: Maximum chunk length
            chunk_overlap: Chunk overlap
            max_chunks_per_doc: Maximum chunks per document
            **kwargs: Additional arguments

        Returns:
            Union[Dict[str, List[str]], List[str]]: Reranked documents
        """
        # Use config values if not specified
        batch_size = batch_size or self.config.batch_size
        show_progress_bar = show_progress_bar if show_progress_bar is not None else self.config.show_progress_bar
        normalize = normalize if normalize is not None else self.config.normalize

        if long_documents_split:
            splitter = DocumentSplitter(
                max_length=chunk_max_length, overlap=chunk_overlap, max_chunks=max_chunks_per_doc
            )
            documents = splitter.split_documents(documents)

        if data_collator:
            scores = self.compute_score(
                [(query, doc) for doc in documents],
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize=normalize,
                data_collator=data_collator,
                **kwargs,
            )
        else:
            scores = self.compute_score(
                [(query, doc) for doc in documents],
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize=normalize,
                **kwargs,
            )

        # Sort documents by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, _ in doc_score_pairs]

        if return_dict:
            return {'documents': sorted_docs, 'scores': [score for _, score in doc_score_pairs]}
        return sorted_docs
