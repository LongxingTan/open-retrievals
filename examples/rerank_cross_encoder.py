import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from retrievals import (
    AutoModelForRanking,
    RerankCollator,
    RerankTrainDataset,
    RerankTrainer,
)

transformers.logging.set_verbosity_error()

model_name_or_path: str = "BAAI/bge-reranker-base"
max_length: int = 512
learning_rate: float = 2e-5
batch_size: int = 32
epochs: int = 3
output_dir: str = "./checkpoints"


def train():
    train_dataset = RerankTrainDataset("t2_ranking.jsonl", positive_key="positive", negative_key="negative")
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
        output_dir=output_dir,
        remove_unused_columns=False,
    )
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankCollator(tokenizer, max_length=max_length),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()
    model.save_pretrained(output_dir)


def predict():
    model = AutoModelForRanking.from_pretrained(model_name_or_path=output_dir)

    examples = [
        [
            "在1974年，第一次在东南亚打自由搏击就得了冠军",
            "1982年打赢了日本重炮手雷龙，接着连续三年打败所有日本空手道高手",
        ],
        ["铁砂掌，源于泗水铁掌帮，三日练成，收费六百", "铁布衫，源于福建省以北70公里，五日练成，收费八百"],
    ]
    scores = model.compute_score(examples)
    print(scores)


if __name__ == '__main__':
    train()
    predict()
