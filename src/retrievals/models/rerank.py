import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.retrievals.data.collator import RerankCollator
from src.retrievals.models.pooling import AutoPooling

logger = logging.getLogger(__name__)


class RerankModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        max_length: Optional[int] = None,
        pooling_method='mean',
        loss_fn=None,
        use_fp16: bool = False,
        pretrained: bool = True,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.graident_checkpointing_enable()
        if use_fp16:
            self.model.half()
        self.pooling = AutoPooling(pooling_method)
        num_features = self.backbone.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)
        self.loss_fn = loss_fn

        if max_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_length = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_length = max_length

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=False)
        encoder_layer = outputs.last_hidden_state
        embeddings = self.pooling(encoder_layer, attention_mask)
        return embeddings

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(features).reshape(-1)
        loss_dict = dict()

        loss = None
        if labels is not None:
            if not self.loss_fn:
                logger.warning('loss_fn should be setup')
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            loss = self.loss_fn(logits, labels)
            loss_dict['loss'] = loss
        return logits, loss, loss_dict

    def compute_score(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str],
        data_collator: RerankCollator,
        batch_size: int = 128,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        assert len(text) == len(text_pair), f"Length of text {len(text)} and text_pair {len(text_pair)} should be same"

        with torch.no_grad():
            scores_list: List = []
            for i in range(0, len(text), batch_size):
                text_batch = [{'query': text[i], 'passage': text_pair[i]} for i in range(i, i + batch_size)]
                batch = data_collator(text_batch)
                scores = self.model(batch, return_dict=True).logits.view(-1).float()
                scores = torch.sigmoid(scores)
                scores_list.extend(scores.cpu().numpy().tolist())

        return scores_list

    def rerank(
        self,
        query,
        passages,
        sentences: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        return

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function to save
        """
        return self.save(path)
