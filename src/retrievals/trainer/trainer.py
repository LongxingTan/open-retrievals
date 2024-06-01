import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch.distributed as dist
import torch.nn as nn
from transformers import Trainer

from ..losses import InfoNCE, SimCSE, TripletLoss

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(self, loss_fn: Optional[Callable] = None, negatives_x_device: bool = False, *args, **kwargs):
        super(RetrievalTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self._dist_loss_scale_factor = dist.get_world_size() if negatives_x_device else 1

    def training_step(self, *args):
        return super().training_step(*args) / self._dist_loss_scale_factor

    def compute_loss(self, model: nn.Module, inputs: Any, return_outputs: bool = False, **kwargs):
        # TODO: 直接使用model返回的loss
        query = inputs["query"]
        pos = inputs["positive"]
        query_embeddings = model(query)
        pos_embeddings = model(pos)
        if 'negative' in inputs:
            neg = inputs["negative"]
            neg_embeddings = model(neg)
            loss = TripletLoss()(query_embeddings, pos_embeddings, neg_embeddings)
        else:
            loss = self.loss_fn(query_embeddings, pos_embeddings)
        if not return_outputs:
            return loss
        outputs = dict()
        outputs['query'] = query_embeddings
        outputs['positive'] = pos_embeddings
        if 'negative' in inputs:
            outputs['negative'] = neg_embeddings
        return (loss, outputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


class RerankTrainer(Trainer):
    def __init__(self, loss_fn: Callable = None, **kwargs):
        super().__init__(**kwargs)
        if not loss_fn:
            loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn

    def compute_loss(self, model: nn.Module, inputs, return_outputs=False, **kwargs):
        outputs: dict = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)
