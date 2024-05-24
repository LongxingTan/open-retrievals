import logging
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
)

from ..data.collator import RerankCollator
from .pooling import AutoPooling
from .utils import get_device_name

logger = logging.getLogger(__name__)


class RerankModel(nn.Module):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer=None,
        pooling_method: str = 'mean',
        loss_fn: Union[nn.Module, Callable] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.pooling = AutoPooling(pooling_method)

        num_features = self.model.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)
        self._init_weights(self.classifier)
        self.loss_fn = loss_fn

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
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
            embeddings = self.pooling(hidden_state, attention_mask)
        else:
            hidden_state = outputs.hidden_states[1]
            embeddings = self.pooling(hidden_state, attention_mask)
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
        if input_ids:
            features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        elif inputs:
            features = self.encode(**inputs)
        else:
            raise ValueError("input_ids(tensor) and inputs(dict) can't be empty as the same time")
        logits = self.classifier(features).reshape(-1)

        if return_dict:
            outputs_dict = dict()
            outputs_dict['logits'] = logits

        if labels is not None:
            if not self.loss_fn:
                logger.warning('loss_fn is not setup, use BCEWithLogitsLoss')
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

    @torch.no_grad()
    def compute_score(
        self,
        text_pairs: Union[List[Tuple[str, str]], Tuple[str, str], List[str, str]],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 128,
        max_length: int = 512,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        if isinstance(text_pairs[0], str):
            text_pairs = [text_pairs]

        batch_size = min(batch_size, len(text_pairs))

        if not data_collator:
            data_collator = RerankCollator(tokenizer=self.tokenizer)

        scores_list: List[float] = []
        for i in range(0, len(text_pairs), batch_size):
            text_batch = [{'query': text_pairs[i], 'document': text_pairs[i]} for i in range(i, i + batch_size)]
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
        query: Union[List[str], str],
        document: Union[List[str], str],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = False,
        show_progress_bar: bool = None,
        return_dict: bool = True,
        **kwargs,
    ):
        merge_scores = self.compute_score(
            text=query,
            text_pair=document,
            data_collator=data_collator,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress_bar,
        )

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_document = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_document.append(document[mid])

        if return_dict:
            return {
                'rerank_document': sorted_document,
                'rerank_scores': sorted_scores,
                'rerank_ids': merge_scores_argsort.tolist(),
            }
        else:
            return sorted_document

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

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str] = None,
        loss_type: Literal['classfication'] = 'classfication',
        num_labels=1,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
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

        reranker = cls(model=model, tokenizer=tokenizer, device=device, loss_type=loss_type)
        return reranker
