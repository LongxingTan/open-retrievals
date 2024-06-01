import transformers
from torch import nn
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from retrievals import (
    ColBertCollator,
    RerankCollator,
    RerankDataset,
    RerankModel,
    RerankTrainer,
)
from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss

transformers.logging.set_verbosity_error()

model_name_or_path: str = "microsoft/deberta-v3-base"
max_length: int = 128
learning_rate: float = 3e-5
batch_size: int = 4
epochs: int = 3

train_dataset = RerankDataset('./t2rank_100.json', positive_key='pos', negative_key='neg')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = RerankModel.from_pretrained(
    model_name_or_path,
    pooling_method="mean",
    loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
)
model = model.set_model_type('colbert')

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir='./checkpoints',
    remove_unused_columns=False,
    logging_steps=100,
)
trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=ColBertCollator(
        tokenizer, query_max_length=max_length, document_max_length=max_length, positive_key='document'
    ),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
