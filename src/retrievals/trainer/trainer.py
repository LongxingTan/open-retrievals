import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from src.retrievals.losses.triplet import TripletLoss

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query = inputs["query"]
        pos = inputs["pos"]
        query_embeddings = model(query)
        pos_embeddings = model(pos)
        if 'neg' in inputs:
            neg = inputs["neg"]
            neg_embeddings = model(neg)
            loss = TripletLoss()(query_embeddings, pos_embeddings, neg_embeddings)
        else:
            loss = self.loss_fn(query_embeddings, pos_embeddings)
        if not return_outputs:
            return loss
        outputs = dict()
        outputs['query'] = query_embeddings
        outputs['pos'] = pos_embeddings
        if 'neg' in inputs:
            outputs['neg'] = neg_embeddings
        return (loss, outputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


class RerankTrainer(Trainer):
    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        if not loss_fn:
            loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, **kwargs):
        return

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass
