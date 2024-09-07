"""Pairwise sentence embedding fine-tuning"""

import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from src.retrievals import AutoModelForEmbedding, PairCollator, RetrievalTrainer
from src.retrievals.losses import InfoNCE, SimCSE, TripletLoss

model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
batch_size: int = 32
epochs: int = 3
output_dir: str = './checkpoints'


def train():
    train_dataset = load_dataset('shibing624/nli_zh', 'STS-B')['train']
    train_dataset = train_dataset.rename_columns({'sentence1': 'query', 'sentence2': 'positive'})
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")
    model = model.set_train_type('pairwise')

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_train_steps = int(len(train_dataset) / batch_size * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
    )
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PairCollator(tokenizer, query_max_length=128, document_max_length=128),
        loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()

    model.save_pretrained(training_args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


def predict():
    model = AutoModelForEmbedding.from_pretrained(output_dir, pooling_method="cls")
    sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
    embeddings = model.encode(sentences)
    print(embeddings)


if __name__ == '__main__':
    train()
    predict()
