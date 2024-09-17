"""LLM Reranker fine-tuning"""

from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from retrievals import (
    LLMRanker,
    LLMRerankCollator,
    RerankTrainer,
    RetrievalTrainDataset,
)
from retrievals.losses import TokenLoss

model_name_or_path: str = "Qwen/Qwen2-1.5B-Instruct"
max_length: int = 256
learning_rate: float = 1e-5
batch_size: int = 8
epochs: int = 3
task_prompt: str = (
    """Given a query A and a passage B, determine whether the passage contains an answer to the query"""
    """by providing a prediction of either 'Yes' or 'No'."""
)


def train():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    train_dataset = RetrievalTrainDataset(
        data_name_or_path='C-MTEB/T2Reranking',
        positive_key='positive',
        negative_key='negative',
        query_instruction='A: ',
        document_instruction='B: ',
        dataset_split='dev',
    )
    data_collator = LLMRerankCollator(
        tokenizer=tokenizer, max_length=max_length, prompt=task_prompt, add_target_token='Yes'
    )
    token_index = tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]
    model = LLMRanker.from_pretrained(
        model_name_or_path,
        causal_lm=True,
        use_fp16=True,
        loss_fn=TokenLoss(token_index=token_index),
        use_lora=True,
    )

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
        logging_steps=100,
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


def predict():
    model_name = 'BAAI/bge-reranker-v2-gemma'

    model = LLMRanker.from_pretrained(
        model_name,
        causal_lm=True,
        use_fp16=True,
    )
    scores = model.compute_score(
        [
            ['what is panda?', 'hi'],
            [
                'what is panda?',
                'The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.',
            ],
        ]
    )
    print(scores)


if __name__ == '__main__':
    train()
    predict()
