"""Pointwise reranking; the trainer owns the configured loss."""

from transformers import Trainer


class CrossEncoderTrainer(Trainer):
    def __init__(self, *args, loss_fn, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"].float().reshape(-1)
        labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        features = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**features, return_dict=True)
        loss = self.loss_fn(outputs["logits"].reshape(-1), labels)
        return (loss, outputs) if return_outputs else loss
