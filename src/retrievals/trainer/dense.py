"""Dense retrieval training with explicit embedding and loss inputs."""

import torch
from transformers import Trainer


class DenseTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        # Retrieval batches have no labels column, but still have an evaluation loss.
        self.can_return_loss = True
        self.label_names = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query, positive, negative = model(inputs)
        loss = self.loss_fn(query, positive, negative)
        if return_outputs:
            return loss, {"scores": query @ positive.T}
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        evaluator = getattr(self, "retrieval_evaluator", None)
        if evaluator is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics = evaluator(self.model, self.args.per_device_eval_batch_size, metric_key_prefix)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Dataset losses remain valid in eval mode and do not require relevance labels.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None
