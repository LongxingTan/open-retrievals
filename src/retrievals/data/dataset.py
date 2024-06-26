import logging
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import datasets
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_name_or_path: Union[str, datasets.Dataset, None] = None,
        train_group_size: int = 2,
        unfold_each_positive: bool = False,
        query_key: str = 'query',
        positive_key: str = 'positive',
        negative_key='negative',
        query_instruction: str = '',
        document_instruction: str = '',
        args: Optional = None,
        tokenizer: PreTrainedTokenizer = None,
    ):
        if not data_name_or_path and args:
            data_name_or_path = args.train_data
        if args and 'train_group_size' in args.__dataclass_fields__:
            self.train_group_size = args.train_group_size
        else:
            self.train_group_size = train_group_size

        if isinstance(data_name_or_path, datasets.Dataset):
            dataset = data_name_or_path
        elif os.path.isdir(data_name_or_path):
            train_datasets = []
            for file in os.listdir(data_name_or_path):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(data_name_or_path, file),
                )

                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)
        else:
            dataset = datasets.load_dataset("json", data_files=data_name_or_path)

        if 'train' in dataset:
            dataset = dataset['train']

        self.unfold_each_positive = unfold_each_positive
        self.tokenizer = tokenizer
        self.args = args
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        args_query_instruction = None
        if self.args and self.args.query_instruction is not None:
            args_query_instruction = self.args.query_instruction
        self.query_instruction = query_instruction if args_query_instruction is None else args_query_instruction
        args_document_instruction = None
        if self.args and self.args.document_instruction is not None:
            args_document_instruction = self.args.document_instruction
        self.document_instruction = (
            document_instruction if args_document_instruction is None else args_document_instruction
        )
        logger.info("Load original {} retrieval data.".format(len(dataset)))

        if self.unfold_each_positive:
            self.samples = self.generate_unfold_samples(dataset)
        else:
            self.dataset = dataset
        logger.info(
            "Generate total {} retrieval data. Query instruction: {}, Document instruction: {}".format(
                len(self.dataset), self.query_instruction, self.document_instruction
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Union[Dict[str, str], List[BatchEncoding]]:
        if self.unfold_each_positive:
            return self.samples[item]

        data = self.dataset[item]
        query = self.query_instruction + data[self.query_key]

        if isinstance(data[self.positive_key], (list, tuple)):
            pos = self.document_instruction + random.choice(data[self.positive_key])
        else:
            pos = self.document_instruction + data[self.positive_key]

        sample = {self.query_key: query, self.positive_key: pos}
        if self.negative_key in data:
            if isinstance(data[self.negative_key], (list, tuple)):
                if len(data[self.negative_key]) < self.train_group_size - 1:
                    num = math.ceil((self.train_group_size - 1) / len(data[self.negative_key]))
                    negs = random.sample(data[self.negative_key] * num, self.train_group_size - 1)
                else:
                    negs = random.sample(data[self.negative_key], self.train_group_size - 1)

            else:
                negs = data[self.negative_key][0]
            sample.update({self.negative_key: [self.document_instruction + neg for neg in negs]})
        return sample

    def generate_unfold_samples(self, dataset):
        samples: List = []
        for data in dataset:
            for pos_text in data[self.positive_key]:
                sample = {
                    self.query_key: self.query_instruction + data[self.query_key],
                    self.positive_key: self.document_instruction + pos_text,
                }

                if self.negative_key in data:
                    if isinstance(data[self.negative_key], (list, tuple)):
                        if len(data[self.negative_key]) < self.train_group_size - 1:
                            num = math.ceil((self.train_group_size - 1) / len(data[self.negative_key]))
                            negs = random.sample(data[self.negative_key] * num, self.train_group_size - 1)
                        else:
                            negs = random.sample(data[self.negative_key], self.train_group_size - 1)
                    else:
                        negs = data[self.negative_key]
                    sample.update({self.negative_key: [self.document_instruction + neg for neg in negs]})
                samples.append(sample)

        return samples

    def dynamic_sample(self, batch_size: int, missing_list=None, wrong_dict=None, max_wrong: int = 16):
        logger.info('Dynamic Shuffle Sample')
        return


class RerankDataset(Dataset):
    def __init__(
        self,
        data_name_or_path: Optional[str] = None,
        unfold_each_positive: bool = False,
        train_group_size: int = 2,
        query_key: str = 'query',
        positive_key: Optional[str] = 'document',
        negative_key: Optional[str] = 'negative',
        args: Optional = None,
        tokenizer: PreTrainedTokenizer = None,
    ):
        """
        train_group_size = 1(positive) + max_negative_samples
        """
        if not data_name_or_path and args:
            data_name_or_path = args.train_data
        if args and 'train_group_size' in args.__dataclass_fields__:
            self.train_group_size = args.train_group_size
        else:
            self.train_group_size = train_group_size

        if args:
            self.query_key = args.query_key or query_key
            self.positive_key = args.positive_key or positive_key
            self.negative_key = args.negative_key or negative_key
            self.unfold_each_positive = args.unfold_each_positive or unfold_each_positive
        else:
            self.query_key = query_key
            self.positive_key = positive_key
            self.negative_key = negative_key
            self.unfold_each_positive = unfold_each_positive

        if isinstance(data_name_or_path, datasets.Dataset):
            dataset = data_name_or_path
        elif os.path.isdir(data_name_or_path):
            train_datasets = []
            for file in os.listdir(data_name_or_path):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(data_name_or_path, file),
                )

                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)

        else:
            dataset = datasets.load_dataset("json", data_files=data_name_or_path)

        if 'train' in dataset:
            dataset = dataset['train']

        logger.info("Load original {} rerank data.".format(len(dataset)))
        if positive_key:
            dataset = self.generate_samples(dataset)

        self.dataset = dataset
        logger.info("Generate total {} rerank data.".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int):
        if isinstance(self.dataset[item], dict):
            query = self.dataset[item][self.query_key]
            document = self.dataset[item][self.positive_key]
            labels = self.dataset[item]['labels']
        else:
            query, document, labels = self.dataset[item]
        sample = {"query": query, "document": document, "labels": labels}
        return sample

    def generate_samples(self, dataset):
        max_negative_samples = None
        if self.train_group_size and self.train_group_size > 0:
            max_negative_samples = self.train_group_size - 1

        samples: List = []
        for data in dataset:
            if self.unfold_each_positive:
                for pos_text in data[self.positive_key]:
                    samples.append([data[self.query_key], pos_text, 1])
            else:
                samples.append([data[self.query_key], random.choice(data[self.positive_key]), 1])

            negative_samples = data[self.negative_key]
            if max_negative_samples:
                if len(negative_samples) < max_negative_samples:
                    num = math.ceil(max_negative_samples / len(negative_samples))
                    negative_samples = random.sample(negative_samples * num, max_negative_samples)
                else:
                    negative_samples = random.sample(negative_samples, max_negative_samples)
            for neg_text in negative_samples:
                samples.append([data[self.query_key], neg_text, 0])
        return samples
