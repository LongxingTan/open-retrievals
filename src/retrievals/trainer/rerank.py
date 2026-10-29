"""Pointwise reranking; the trainer owns the configured loss."""

import torch
import torch.nn.functional as F
from transformers import Trainer


class CrossEncoderTrainer(Trainer):
    def __init__(self, *args, loss_fn, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"].float().reshape(-1)
        labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        features = {key: value for key, value in inputs.items() if key not in {"labels", "group_sizes"}}
        outputs = model(**features, return_dict=True)
        logits = outputs["logits"].reshape(-1)
        if "group_sizes" in inputs:
            groups = logits.split(inputs["group_sizes"].tolist())
            loss = torch.stack(
                [
                    F.cross_entropy(
                        group.unsqueeze(0),
                        torch.zeros(1, dtype=torch.long, device=group.device),
                        label_smoothing=self.label_smoothing,
                    )
                    for group in groups
                ]
            ).mean()
        else:
            loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
