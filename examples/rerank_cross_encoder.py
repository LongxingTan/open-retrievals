import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import evaluate
import torch
import transformers
from peft import LoraConfig
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
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
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    # cache_dir: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Where do you want to store the pretrained models downloaded from s3"
    #     },
    # )


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to train data"})
    train_group_size: int = field(default=8)
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
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
    passage_instruction: str = field(default=None, metadata={"help": "instruction for passage"})

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
    use_lora: bool = field(default=True)


class RerankTrainingDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return

    def __getitem__(self, item):
        return


def get_optimizer():
    return


def get_scheduler():
    return


def compute_metrics(eval_pred):
    return


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

    tokenizer = AutoTokenizer.from_pretrained()

    train_dataset = RerankTrainingDataset(args=data_args, tokenizer=tokenizer)
    eval_dataset = RerankTrainingDataset()

    model = RerankModel()
    optimizer = get_optimizer()
    scheduler = get_scheduler()

    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=RerankCollator(),
        compute_metrics=compute_metrics,
    )
    trainer.optimzier = optimizer
    trainer.scheduler = scheduler

    eval_res = trainer.evaluate()
    print(f"{eval_res=}")
    trainer.train()
