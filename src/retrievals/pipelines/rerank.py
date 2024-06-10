import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import transformers
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from ..data import ColBertCollator, RerankCollator, RerankDataset, RetrievalDataset
from ..models.rerank import AutoModelForRanking, ColBERT
from ..trainer import RerankTrainer

transformers.logging.set_verbosity_error()
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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to corpus"})
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
    passage_instruction: str = field(default=None, metadata={"help": "instruction for passage"})

    # def __post_init__(self):
    #     if not os.path.exists(self.train_data):
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class RerankerTrainingArguments(TrainingArguments):
    model_type: str = field(default='cross-encoder', metadata={'help': "train type of cross-encoder, colbert"})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    remove_unused_columns: bool = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) is not empty. Use --overwrite_output_dir to overcome."
        )

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
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    if training_args.model_type == 'colbert':
        logger.info('Set model to ColBERT')
        train_dataset = RetrievalDataset(
            args=data_args,
            tokenizer=tokenizer,
            positive_key=data_args.positive_key,
            negative_key=data_args.negative_key,
        )
        data_collator = ColBertCollator(
            tokenizer,
            query_max_length=128,
            document_max_length=data_args.max_length,
            positive_key=data_args.positive_key,
            negative_key=data_args.negative_key,
        )
        model = ColBERT.from_pretrained(model_args.model_name_or_path, colbert_dim=128)
    else:
        logger.info('Set model to CrossEncoder')
        train_dataset = RerankDataset(args=data_args, tokenizer=tokenizer)
        data_collator = RerankCollator(tokenizer, max_length=data_args.max_length)
        model = AutoModelForRanking.from_pretrained(
            model_args.model_name_or_path, num_labels=1, loss_fn=nn.BCEWithLogitsLoss(reduction='mean')
        )

    logger.info(f"Total examples for training: {len(train_dataset)}")
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    # trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
