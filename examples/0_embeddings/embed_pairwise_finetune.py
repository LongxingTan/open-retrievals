import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import datasets
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from retrievals import (
    AutoModelForEmbedding,
    AutoModelForMatch,
    RetrievalTrainer,
    TripletCollator,
)
from retrievals.losses import TripletLoss

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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )


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
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "the max number of examples for each dataset"},
    )

    query_instruction: str = field(default=None, metadata={"help": "instruction for query"})
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
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(default="cls", metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
    # save_steps


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.train_data, file),
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
            self.dataset = datasets.load_dataset("json", data_files=args.train_data, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        query = self.dataset[item]["query"]
        pos = self.dataset[item]["pos"][0]
        neg = self.dataset[item]["neg"][0]
        # pos = random.choice(self.dataset[item]["pos"])
        # neg = random.choice(self.dataset[item]["neg"])

        res = {"query": query, "pos": pos, "neg": neg}
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
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

    # num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name
    #     if model_args.config_name
    #     else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    # model = PairwiseModel(model_args.model_name_or_path, pooling_method="mean")
    model = AutoModelForEmbedding.from_pretrained(model_args.model_name_or_path, pooling_method="mean")
    optimizer = get_optimizer(model, lr=5e-5, weight_decay=1e-3)

    # TODO: total steps更新
    lr_scheduler = get_scheduler(optimizer, num_train_steps=int(len(train_dataset) / 2 * 1))

    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=TripletCollator(tokenizer, max_length=data_args.query_max_len),
        loss_fn=TripletLoss(),
        # optimizers=(optimizer, lr_scheduler),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = lr_scheduler
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
