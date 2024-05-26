import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..losses import InfoNCE, SimCSE, TripletLoss

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


class RerankTrainer(Trainer):
    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        if not loss_fn:
            loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs: dict = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)
