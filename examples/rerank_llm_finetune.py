from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AdamW,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from retrievals import (
    AutoModelForRanking,
    LLMRerankCollator,
    RerankTrainDataset,
    RerankTrainer,
    RetrievalTrainDataset,
)
from retrievals.losses import TokenLoss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    causal_lm: bool = field(default=False, metadata={'help': "Whether the model is a causal lm or not"})


@dataclass
class DataArguments:
    data_name_or_path: str = field(default=None, metadata={"help": "Path to corpus"})
    train_group_size: int = field(default=8)
    unfold_each_positive: bool = field(default=False)
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    query_key: str = field(default=None)
    positive_key: str = field(default=None)
    negative_key: str = field(default=None)

    query_instruction: str = field(default=None, metadata={"help": "instruction for query"})
    document_instruction: str = field(default=None, metadata={"help": "instruction for document"})
    task_prompt: str = field(
        default=(
            "Given a query A and a passage B, determine whether the passage contains an answer "
            "to the query by providing a prediction of either 'Yes' or 'No'."
        )
    )


@dataclass
class RerankerTrainingArguments(TrainingArguments):
    model_type: str = field(default='cross-encoder', metadata={'help': "train type of cross-encoder, colbert"})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    use_inbatch_negative: bool = field(default=False)
    temperature: Optional[float] = field(default=0.02)
    remove_unused_columns: bool = field(default=False)
    num_train_epochs: int = field(default=3)
    use_lora: bool = field(default=False)
    use_bnb_config: bool = field(default=False)
    do_rerank: bool = field(default=False, metadata={"help": "run the reranking loop"})


def get_optimizer(model, learning_rate, weight_decay=0.0):
    optimizer_parameters = [
        {
            "params": [p for n, p in model.model.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": learning_rate * 20,
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_parameters)


parser = HfArgumentParser((ModelArguments, DataArguments, RerankerTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=False,
)

train_dataset = RetrievalTrainDataset(
    args=data_args,
    tokenizer=tokenizer,
    unfold_each_positive=data_args.unfold_each_positive,
    train_group_size=data_args.train_group_size,
    positive_key=data_args.positive_key,
    negative_key=data_args.negative_key,
)
data_collator = LLMRerankCollator(tokenizer=tokenizer, max_length=data_args.max_length, prompt=data_args.task_prompt)
token_index = tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]
model = AutoModelForRanking.from_pretrained(
    model_args.model_name_or_path,
    num_labels=1,
    loss_fn=TokenLoss(token_index=token_index, train_group_size=data_args.train_group_size),
    causal_lm=True,
    use_lora=training_args.use_lora,
    quantization_config=None,
)
optimizer = get_optimizer(model, learning_rate=training_args.learning_rate)

num_train_steps = int(len(train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
)

trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler

trainer.train()
model.save_pretrained(training_args.output_dir)
