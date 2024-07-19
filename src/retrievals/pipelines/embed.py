"""Embedding fine tune pipeline"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from ..data import (
    EncodeCollator,
    EncodeDataset,
    PairCollator,
    RetrievalTrainDataset,
    TripletCollator,
)
from ..losses import InfoNCE, SimCSE, TripletLoss
from ..models.embedding_auto import AutoModelForEmbedding
from ..trainer import RetrievalTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
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
    lora_path: Optional[str] = field(default=None, metadata={'help': "Lora adapter save path"})


@dataclass
class DataArguments:
    data_name_or_path: str = field(default=None, metadata={"help": "Path to train data"})
    train_group_size: int = field(default=2)
    unfold_each_positive: bool = field(default=False)
    query_max_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    document_max_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    query_instruction: str = field(default=None, metadata={"help": "instruction for query"})
    document_instruction: str = field(default=None, metadata={"help": "instruction for document"})
    query_key: str = field(default=None)
    positive_key: str = field(default='positive')
    negative_key: str = field(default='negative')
    is_query: bool = field(default=False)
    encoding_save_file: str = field(default='embed.pkl')

    def __post_init__(self):
        if self.data_name_or_path is not None:
            info = self.data_name_or_path.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.data_name_or_path = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.data_name_or_path:
                self.data_name_or_path, self.dataset_language = self.data_name_or_path.split(':')
        else:
            self.data_name_or_path = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    train_type: str = field(default='pairwise', metadata={'help': "train type of point, pair, or list"})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    loss_fn: str = field(default='infonce')
    use_inbatch_negative: bool = field(default=True, metadata={"help": "use documents in the same batch as negatives"})
    remove_unused_columns: bool = field(default=False)
    use_lora: bool = field(default=False)
    use_bnb_config: bool = field(default=False)
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RetrieverTrainingArguments))
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
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
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

    if training_args.do_train:
        model = AutoModelForEmbedding.from_pretrained(
            model_name_or_path=model_args.model_name_or_path,
            pooling_method=training_args.pooling_method,
            use_lora=training_args.use_lora,
            quantization_config=quantization_config,
        )

        if training_args.loss_fn == 'infonce':
            loss_fn = InfoNCE(
                nn.CrossEntropyLoss(label_smoothing=0.0),
                use_inbatch_negative=training_args.use_inbatch_negative,
                temperature=training_args.temperature,
            )
        elif training_args.loss_fn == 'simcse':
            loss_fn = SimCSE(temperature=training_args.temperature)
        elif training_args.loss_fn == 'triplet':
            loss_fn = (TripletLoss(training_args.temperature),)
        else:
            raise ValueError

        model = model.set_train_type(
            "pairwise",
            loss_fn=loss_fn,
        )

        train_dataset = RetrievalTrainDataset(
            args=data_args,
            tokenizer=tokenizer,
            positive_key=data_args.positive_key,
            negative_key=data_args.negative_key,
        )
        logger.info(f"Total training examples: {len(train_dataset)}")

        trainer = RetrievalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=TripletCollator(
                tokenizer,
                query_max_length=data_args.query_max_length,
                document_max_length=data_args.document_max_length,
                positive_key=data_args.positive_key,
                negative_key=data_args.negative_key,
            ),
        )

        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

        trainer.train()
        # trainer.save_model(training_args.output_dir)
        model.save_pretrained(training_args.output_dir)

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_encode:
        model = AutoModelForEmbedding.from_pretrained(
            model_name_or_path=model_args.model_name_or_path,
            pooling_method=training_args.pooling_method,
            use_lora=training_args.use_lora,
            quantization_config=quantization_config,
            lora_path=model_args.lora_path,
        )

        max_length = data_args.query_max_length if data_args.is_query else data_args.document_max_length
        logger.info(f'Encoding will be saved in {training_args.output_dir}')

        encode_dataset = EncodeDataset(args=data_args, tokenizer=tokenizer, max_length=max_length, text_key='text')
        logger.info(f"Number of train samples: {len(encode_dataset)}")

        encode_loader = DataLoader(
            encode_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=EncodeCollator(tokenizer, max_length=max_length, padding='max_length'),
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        # embeddings = []
        # lookup_indices = []
        # for batch_ids, batch in tqdm(encode_loader):
        #     lookup_indices.extend(batch_ids)
        #     embed = model.encode(batch)
        #     embeddings.append(embed)
        # embeddings = np.concatenate(embeddings)

        embeddings = model.encode(encode_loader)
        lookup_indices = list(range(len(encode_dataset)))

        with open(os.path.join(training_args.output_dir, data_args.encoding_save_file), 'wb') as f:
            pickle.dump((embeddings, lookup_indices), f)


if __name__ == "__main__":
    main()
