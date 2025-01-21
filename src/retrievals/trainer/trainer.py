import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import device
from transformers import Trainer

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(
        self,
        loss_fn: Optional[Callable] = None,
        train_type: str = 'normal',
        negatives_cross_device: bool = False,
        *args,
        **kwargs,
    ):
        super(RetrievalTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.train_type = train_type
        self.negative_cross_device = negatives_cross_device
        self._dist_loss_scale_factor = dist.get_world_size() if negatives_cross_device else 1

    def training_step(self, *args):
        return super().training_step(*args) / self._dist_loss_scale_factor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.train_type == 'pairwise':
            return self.compute_pair_loss(model=model, inputs=inputs, return_outputs=return_outputs)

        outputs = model(inputs, return_dict=True)
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            return self.loss_fn(*outputs, negatives_cross_device=self.negative_cross_device)

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

        loss = self.loss_fn(
            query_embeddings, pos_embeddings, neg_embeddings, negatives_cross_device=self.negatives_cross_device
        )
        if not return_outputs:
            return loss

        outputs = dict()
        outputs['query'] = query_embeddings
        outputs['positive'] = pos_embeddings
        if 'negative' in inputs:
            outputs['negative'] = neg_embeddings
        return loss, outputs

    def evaluate(self, eval_dataset=None, metrics=None, **kwargs):
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided.")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_embeddings = []
        all_labels = []

        logger.info("Evaluating model...")
        self.model.eval()

        for batch in eval_dataloader:
            with torch.no_grad():
                for key, v in batch.items():
                    embeddings = self.model(v.to(self.args.device))
                    all_embeddings.append(embeddings)

                if "labels" in batch:
                    all_labels.append(batch["labels"])

        all_query_embeddings = torch.cat(all_embeddings[::2], dim=0)
        all_doc_embeddings = torch.cat(all_embeddings[1::2], dim=0)
        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
            results = metrics(all_query_embeddings, all_doc_embeddings, all_labels)
        else:
            results = metrics(all_query_embeddings, all_doc_embeddings)

        logger.info(f"Evaluation results: {results}")
        return results

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving retrieval model checkpoint to {output_dir}")
        save_model = self.model.model if hasattr(self.model, 'model') else self.model
        save_model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


class RerankTrainer(Trainer):
    def __init__(self, loss_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        if loss_fn is None:
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


class DistillTrainer(Trainer):
    # https://github.com/texttron/tevatron/blob/tevatron-v1/src/tevatron/distillation/trainer.py
    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 1.0,
        **kwargs,
    ):
        super(DistillTrainer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self._dist_loss_scale_factor = 1

    def compute_loss(self, model: nn.Module, inputs, return_outputs=False, **kwargs):
        student_outputs = model(**inputs)
        student_scores = student_outputs.logits

        with torch.no_grad():
            teacher_scores = self.teacher_model(**inputs).logits
            teacher_probs = self._get_teacher_probabilities(teacher_scores, student_scores.shape)

        student_log_probs = nn.functional.log_softmax(student_scores / self.temperature, dim=1)
        loss = (
            nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * self._dist_loss_scale_factor
        )
        return (loss, student_outputs) if return_outputs else loss

    @torch.no_grad()
    def _get_teacher_probabilities(self, teacher_scores: torch.Tensor, student_shape):
        batch_size = student_shape[0]
        teacher_mat = torch.zeros(student_shape, dtype=teacher_scores.dtype, device=teacher_scores.device)
        teacher_scores = torch.softmax(teacher_scores.view(batch_size, -1) / self.temperature, dim=1)
        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device).view(batch_size, -1)
        return torch.scatter(teacher_mat, dim=-1, index=index, src=teacher_scores)
