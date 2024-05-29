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
from torch import nn
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
    gradient_accumulation_steps: int = 1024
    fp16: bool = True


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
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    model = RerankModel.from_pretrained(
        model_args.model_name_or_path,
        pooling_method=model_args.pooling_method,
        loss_fn=loss_fn,
    )
    model.set_model_type('colbert')
