import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from retrievals import RerankCollator, RerankModel, RerankTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    pooling_method: str = field(default="mean")


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to train data"})
    train_group_size: int = field(default=8)
    query_max_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    document_max_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "the max number of examples for each dataset"},
    )
    query_instruction: str = field(
        default="Instruct: Retrieve semantically similar text.\nQuery: ", metadata={"help": "instruction for query"}
    )
    document_instruction: str = field(default=None, metadata={"help": "instruction for document"})

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_train: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    remove_unused_columns: bool = False
    cache_dir: Optional[str] = field(default="/root/autodl-tmp/llm_output")
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(default="cls", metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
    gradient_accumulation_steps: int = 1024
    fp16: bool = True


class RerankTrainingDataset(Dataset):
    def __init__(self, df, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        dataset = datasets.Dataset.from_pandas(df)
        self.tokenizer = tokenizer
        data_dict = dataset.map(partial(preprocess, tokenizer=tokenizer, max_length=max_length), batched=True)

        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[item],
            attention_mask=self.attention_mask[item],
            labels=self.labels[item],
        )


def preprocess(examples, tokenizer, max_length):
    tokenized_text = tokenizer(
        examples["query"],
        examples["document"],
        padding="max_length",
        truncation="longest_first",
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    tokenized_text["labels"] = torch.tensor(examples["labels"]).long()
    return tokenized_text


def get_optimizer():
    return


def get_scheduler():
    return


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(labels, predictions)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            extra_args,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    df = pd.read_csv(data_args.train_data)

    train_dataset = RerankTrainingDataset(df=df, tokenizer=tokenizer, max_length=data_args.query_max_len)
    # eval_dataset = RerankTrainingDataset()

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    model = RerankModel.from_pretrained(
        model_args.model_name_or_path,
        pooling_method=model_args.pooling_method,
        loss_fn=loss_fn,
    )
    # optimizer = get_optimizer()
    # scheduler = get_scheduler()

    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        # data_collator=RerankCollator(),
        compute_metrics=compute_metrics,
    )
    # trainer.optimizer = optimizer
    # trainer.scheduler = scheduler

    # eval_res = trainer.evaluate()
    # print(f"{eval_res=}")
    trainer.train()


if __name__ == "__main__":
    main()
