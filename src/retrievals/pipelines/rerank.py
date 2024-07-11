"""Reranking fine tune pipeline"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch import nn
from transformers import (
    AdamW,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
)

from ..data import (
    ColBertCollator,
    RerankCollator,
    RerankTrainDataset,
    RetrievalTrainDataset,
)
from ..data.collator import LLMRerankCollator
from ..losses import ColbertLoss, TokenLoss
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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

    if training_args.use_bnb_config:
        from transformers import BitsAndBytesConfig

        logger.info('Use quantization bnb config')
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    if training_args.model_type == 'colbert':
        logger.info('Set rank model to ColBERT')
        train_dataset = RetrievalTrainDataset(
            args=data_args,
            tokenizer=tokenizer,
            train_group_size=data_args.train_group_size,
            unfold_each_positive=data_args.unfold_each_positive,
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
        model = ColBERT.from_pretrained(
            model_args.model_name_or_path,
            colbert_dim=512,
            loss_fn=ColbertLoss(use_inbatch_negative=training_args.use_inbatch_negative),
        )
    elif training_args.model_type == 'cross-encoder':
        logger.info('Set rank model to CrossEncoder')
        train_dataset = RerankTrainDataset(args=data_args, tokenizer=tokenizer)
        data_collator = RerankCollator(tokenizer, max_length=data_args.max_length)
        model = AutoModelForRanking.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            loss_fn=nn.BCEWithLogitsLoss(reduction='mean'),
            causal_lm=model_args.causal_lm,
        )
    elif training_args.model_type == 'llm':
        logger.info('Set rank model to LLM')
        train_dataset = RetrievalTrainDataset(
            args=data_args,
            tokenizer=tokenizer,
            unfold_each_positive=data_args.unfold_each_positive,
            train_group_size=data_args.train_group_size,
            positive_key=data_args.positive_key,
            negative_key=data_args.negative_key,
        )
        data_collator = LLMRerankCollator(
            tokenizer=tokenizer, max_length=data_args.max_length, prompt=data_args.task_prompt
        )
        token_index = tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]
        model = AutoModelForRanking.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            loss_fn=TokenLoss(token_index=token_index, train_group_size=data_args.train_group_size),
            causal_lm=True,
            use_lora=training_args.use_lora,
            quantization_config=quantization_config,
        )
    else:
        raise ValueError(
            f'model_type should be one of colbert, cross-encoder and llm, instead of {training_args.model_type}'
        )

    logger.info(f"Total training examples: {len(train_dataset)}")
    optimizer = get_optimizer(model, learning_rate=training_args.learning_rate)

    num_train_steps = int(
        len(train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs
    )
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

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    # trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
