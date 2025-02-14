"""LLM sentence embedding fine-tuning"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from retrievals import (
    AutoModelForEmbedding,
    AutoModelForRetrieval,
    PairwiseModel,
    RetrievalCollator,
    RetrievalTrainer,
)
from retrievals.losses import InfoNCE, TripletLoss, TripletRankingLoss

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="intfloat/e5-mistral-7b-instruct",
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


@dataclass
class DataArguments:
    data_name_or_path: str = field(default="Tevatron/scifact", metadata={"help": "Path to train data"})
    train_group_size: int = field(default=8)
    query_max_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    document_max_length: int = field(
        default=64,
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
        self.dataset_split = 'train'
        self.dataset_language = 'default'

        if self.data_name_or_path is not None:
            if not os.path.isfile(self.data_name_or_path) and not os.path.isdir(self.data_name_or_path):
                info = self.data_name_or_path.split('/')
                self.dataset_split = info[-1] if len(info) == 3 else 'train'
                self.data_name_or_path = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
                self.dataset_language = 'default'
                if ':' in self.data_name_or_path:
                    self.data_name_or_path, self.dataset_language = self.data_name_or_path.split(':')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_train: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    remove_unused_columns: bool = False
    cache_dir: Optional[str] = None
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    pooling_method: str = field(default="cls", metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    gradient_accumulation_steps: int = field(default=1)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=100)
    output_dir: str = field(default='./checkpoint')
    save_total_limit: int = field(default=1)


@dataclass
class LoraArguments:
    lora_alpha: int = (32,)
    lora_dropout: float = 0.1
    peft_type: str = "LORA"
    r: int = (16,)

    target_modules = (["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],)
    task_type = "FEATURE_EXTRACTION"


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if os.path.isdir(args.data_name_or_path):
            train_datasets = []
            for file in os.listdir(args.data_name_or_path):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.data_name_or_path, file),
                    split="train",
                )
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(
                            list(range(len(temp_dataset))),
                            args.max_example_num_per_dataset,
                        )
                    )
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            # self.dataset = datasets.load_dataset("json", data_files=args.data_name_or_path, split="train")
            self.dataset = datasets.load_dataset(args.data_name_or_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        query = self.dataset[item]["query"] + self.tokenizer.eos_token
        pos = self.dataset[item]["pos"][0] + self.tokenizer.eos_token
        neg = self.dataset[item]["neg"][0] + self.tokenizer.eos_token
        res = {"query": query, "positive": pos, "negative": neg}
        return res


def get_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    no_decay_keywords: Sequence[str] = ("bias", "LayerNorm", "layernorm"),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_scheduler(optimizer, num_train_steps, num_warmup_steps=None):
    if not num_warmup_steps:
        num_warmup_steps = num_train_steps * 0.05
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        # num_cycles=cfg.num_cycles,
    )
    return scheduler


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

    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=training_args.cache_dir,
        use_fast=False,
    )

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    print('Number of samples: ', len(train_dataset))

    lora_config = LoraArguments()

    model = AutoModelForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        pooling_method=training_args.pooling_method,
        lora_config=lora_config,
    )

    train_model = PairwiseModel(
        model,
        loss_fn=TripletLoss(
            use_inbatch_negative=training_args.use_inbatch_neg,
            negatives_cross_device=training_args.negatives_cross_device,
        ),
    )

    optimizer = get_optimizer(train_model, lr=5e-5, weight_decay=1e-3)

    lr_scheduler = get_scheduler(
        optimizer,
        num_train_steps=int(
            len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        ),
    )

    trainer = RetrievalTrainer(
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RetrievalCollator(
            tokenizer,
            keys=['query', 'positive', 'negative'],
            max_lengths=[data_args.query_max_length, data_args.document_max_length, data_args.document_max_length],
        ),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = lr_scheduler
    trainer.train()

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
