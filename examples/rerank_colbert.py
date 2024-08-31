import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from retrievals import ColBERT, ColBertCollator, RerankTrainer, RetrievalTrainDataset
from retrievals.losses import ColbertLoss

transformers.logging.set_verbosity_error()


model_name_or_path: str = "microsoft/deberta-v3-base"
learning_rate: float = 3e-5
batch_size: int = 32
epochs: int = 3
colbert_dim: int = 128
output_dir: str = './checkpoints'


def train():
    train_dataset = RetrievalTrainDataset("t2_ranking.jsonl", positive_key="positive", negative_key="negative")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    data_collator = ColBertCollator(
        tokenizer,
        query_max_length=32,
        document_max_length=128,
        positive_key='positive',
        negative_key='negative',
    )
    model = ColBERT.from_pretrained(
        model_name_or_path,
        colbert_dim=colbert_dim,
        loss_fn=ColbertLoss(use_inbatch_negative=True),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_train_steps = int(len(train_dataset) / batch_size * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
    )

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        output_dir=output_dir,
        remove_unused_columns=False,
    )
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()
    model.save_pretrained(output_dir)


def predict():
    model = ColBERT.from_pretrained(model_name_or_path=output_dir, colbert_dim=colbert_dim)
    examples = [
        [
            "In 1974, I won the championship in Southeast Asia in my first kickboxing match",
            "In 1982, I defeated the heavy hitter Ryu Long.",
        ]
    ]
    scores = model.compute_score(examples)
    print(scores)


if __name__ == '__main__':
    train()
    predict()
