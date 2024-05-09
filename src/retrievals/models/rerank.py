import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ..data.collator import RerankCollator
from .embedding_auto import get_device_name
from .pooling import AutoPooling

logger = logging.getLogger(__name__)


class RerankModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        pooling_method: str = 'mean',
        loss_fn: Union[nn.Module, Callable] = None,
        max_length: Optional[int] = None,
        use_fp16: bool = False,
        use_lora: bool = False,
        lora_config=None,
        gradient_checkpointing: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if gradient_checkpointing:
            self.model.graident_checkpointing_enable()
        if device is None:
            self.device = get_device_name()
        else:
            self.device = device

        if use_fp16:
            self.model.half()
        if use_lora:
            # peft config and wrapping
            from peft import LoraConfig, TaskType, get_peft_model

            if not lora_config:
                raise ValueError("If use_lora is true, please provide a valid lora_config")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.pooling = AutoPooling(pooling_method)
        num_features = self.model.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)
        # self._init_weights(self.classifier)
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
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
            return logits

    def compute_score(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 128,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        assert len(text) == len(text_pair), f"Length of text {len(text)} and text_pair {len(text_pair)} should be same"
        batch_size = min(batch_size, len(text))

        if not data_collator:
            data_collator = RerankCollator(tokenizer=self.tokenizer)

        with torch.no_grad():
            scores_list: List = []
            for i in range(0, len(text), batch_size):
                text_batch = [{'query': text[i], 'document': text_pair[i]} for i in range(i, i + batch_size)]
                batch = data_collator(text_batch)
                scores = (
                    self.model(batch['input_ids'], batch['attention_mask'], return_dict=True).logits.view(-1).float()
                )
                scores = torch.sigmoid(scores)
                scores_list.extend(scores.cpu().numpy().tolist())

        return scores_list

    def rerank(
        self,
        query: Union[List[str], str],
        document: List[str],
        data_collator: Optional[RerankCollator] = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        return_dict: bool = True,
        **kwargs,
    ):
        merge_scores = self.compute_score(query, document, data_collator, batch_size, show_progress_bar)

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
