import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from retrievals import AutoModelForRanking, RerankCollator, RerankDataset, RerankTrainer

transformers.logging.set_verbosity_error()

model_name_or_path: str = "microsoft/deberta-v3-base"
max_length: int = 128
learning_rate: float = 3e-5
batch_size: int = 4
epochs: int = 3

train_dataset = RerankDataset("t2_ranking.jsonl", positive_key="positive", negative_key="negative")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForRanking.from_pretrained(model_name_or_path)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.05 * num_train_steps,
    num_training_steps=num_train_steps,
)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir="./checkpoints",
    remove_unused_columns=False,
)
trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, query_max_length=max_length, document_max_length=max_length),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
