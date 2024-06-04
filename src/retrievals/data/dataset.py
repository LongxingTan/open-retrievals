import copy
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
        query_key: str = 'query',
        positive_key: str = 'positive',
        negative_key='negative',
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
            self.dataset = data_name_or_path
        elif os.path.isdir(data_name_or_path):
            train_datasets = []
            for file in os.listdir(data_name_or_path):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(data_name_or_path, file),
                )

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset("json", data_files=data_name_or_path)

        if 'train' in self.dataset:
            self.dataset = self.dataset['train']

        self.tokenizer = tokenizer
        self.args = args
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        logger.info("Loaded {} retrieval data.".format(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Union[Dict[str, str], List[BatchEncoding]]:
        query = self.dataset[item][self.query_key]
        if self.args and self.args.query_instruction is not None:
            query = self.args.query_instruction + query

        if isinstance(self.dataset[item][self.positive_key], (list, tuple)):
            pos = random.choice(self.dataset[item][self.positive_key])
        else:
            pos = self.dataset[item][self.positive_key]

        sample = {self.query_key: query, self.positive_key: pos}
        if self.negative_key in self.dataset[item]:
            if isinstance(self.dataset[item][self.negative_key], (list, tuple)):
                if len(self.dataset[item][self.negative_key]) < self.train_group_size - 1:
                    num = math.ceil((self.train_group_size - 1) / len(self.dataset[item][self.negative_key]))
                    negs = random.sample(self.dataset[item][self.negative_key] * num, self.train_group_size - 1)
                else:
                    negs = random.sample(self.dataset[item][self.negative_key], self.train_group_size - 1)

            else:
                negs = self.dataset[item][self.negative_key][0]
            sample.update({self.negative_key: negs})
        return sample

    def dynamic_sample(self, batch_size: int, missing_list=None, wrong_dict=None, max_wrong: int = 16):
        logger.info('\nDynamic Shuffle Sample...')

        return

    @classmethod
    def from_local(cls):
        return


class RerankDataset(Dataset):
    def __init__(
        self,
        data_name_or_path: str,
        cache_dir: Optional[str] = None,
        query_key='query',
        positive_key: Optional[str] = None,
        negative_key: Optional[str] = None,
        negative_numbers: Optional[int] = None,
    ):
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.negative_numbers = negative_numbers

        if os.path.isdir(data_name_or_path):
            train_datasets = []
            for file in os.listdir(data_name_or_path):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(data_name_or_path, file),
                    cache_dir=cache_dir,
                )

                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)

        else:
            dataset = datasets.load_dataset("json", data_files=data_name_or_path)

        if 'train' in dataset:
            dataset = dataset['train']

        if positive_key:
            dataset = self.generate_samples(dataset)

        self.dataset = dataset
        logger.info("Loaded {} rerank data.".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int):
        if isinstance(self.dataset[item], dict):
            query = self.dataset[item][self.query_key]
            document = self.dataset[item]['document']
            labels = self.dataset[item]['labels']
        else:
            query, document, labels = self.dataset[item]
        sample = {"query": query, "document": document, "labels": labels}
        return sample

    def generate_samples(self, dataset):
        samples: List = []
        for data in dataset:
            for text_pos in data[self.positive_key]:
                samples.append([data[self.query_key], text_pos, 1])

            negative_samples = data[self.negative_key]
            if self.negative_numbers:
                # TODO: random strategy
                negative_samples = negative_samples[: self.negative_numbers]
            for text_neg in negative_samples:
                samples.append([data[self.query_key], text_neg, 0])
        return samples

    @classmethod
    def from_local(cls):
        return
