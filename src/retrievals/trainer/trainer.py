import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Trainer

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(
        self,
        loss_fn: Optional[Callable] = None,
        train_type: str = 'normal',
        negatives_x_device: bool = False,
        *args,
        **kwargs,
    ):
        super(RetrievalTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.train_type = train_type
        self._dist_loss_scale_factor = dist.get_world_size() if negatives_x_device else 1

    def training_step(self, *args):
        return super().training_step(*args) / self._dist_loss_scale_factor

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.train_type == 'pairwise':
            return self.compute_pair_loss(model=model, inputs=inputs, return_outputs=return_outputs)

        outputs = model(inputs, return_dict=True)
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        else:
            return self.loss_fn(*outputs)

    def compute_pair_loss(self, model: nn.Module, inputs: Any, return_outputs: bool = False):
        query = inputs["query"]
        if 'document' in inputs:
            pos = inputs["document"]
        else:
            pos = inputs['positive']
        query_embeddings = model(query)
        pos_embeddings = model(pos)
        if 'negative' in inputs:
            neg = inputs["negative"]
            neg_embeddings = model(neg)
        else:
            neg_embeddings = None

        loss = self.loss_fn(query_embeddings, pos_embeddings, neg_embeddings)
        if not return_outputs:
            return loss

        outputs = dict()
        outputs['query'] = query_embeddings
        outputs['positive'] = pos_embeddings
        if 'negative' in inputs:
            outputs['negative'] = neg_embeddings
        return loss, outputs

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving retrieval model checkpoint to {output_dir}")
        save_model = self.model.model if hasattr(self.model, 'model') else self.model
        save_model.save_pretrained(output_dir)
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
        logger.info(f"Saving rerank model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


class DistilTrainer(Trainer):
    # https://github.com/texttron/tevatron/blob/tevatron-v1/src/tevatron/distillation/trainer.py
    def __init__(self, teacher_model):
        super(DistilTrainer, self).__init__()
        self.teacher_model = teacher_model
        self._dist_loss_scale_factor = 1

    def compute_loss(self, model, inputs, return_outputs=False):
        student_scores = model(inputs)
        with torch.no_grad():
            teacher_scores = self.teacher_model(inputs)

        teacher_mat = torch.zeros(student_scores.shape, dtype=student_scores.dtype, device=teacher_scores.device)
        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)
        teacher_scores = torch.softmax(
            teacher_scores.view(student_scores.size(0), -1), dim=1, dtype=student_scores.dtype
        )
        teacher_mat = torch.scatter(
            teacher_mat, dim=-1, index=index.view(student_scores.size(0), -1), src=teacher_scores
        )
        student_scores = nn.functional.log_softmax(student_scores, dim=1)
        loss = nn.functional.kl_div(student_scores, teacher_mat, reduction='batchmean') * self._dist_loss_scale_factor
        return loss
