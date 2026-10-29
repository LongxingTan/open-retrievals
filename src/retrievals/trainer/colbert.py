"""Late-interaction training with explicit token masks."""

from .dense import DenseTrainer


class ColBERTTrainer(DenseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, return_embeddings=True)
        loss = self.loss_fn(
            outputs["query"],
            outputs["positive"],
            outputs.get("negative"),
            mask=inputs["query_attention_mask"][:, 1:],
            document_mask=inputs["pos_attention_mask"][:, 1:],
            negative_mask=inputs["neg_attention_mask"][:, 1:] if "neg_attention_mask" in inputs else None,
        )
        return (loss, outputs) if return_outputs else loss
