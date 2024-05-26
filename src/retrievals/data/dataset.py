import copy
import logging
import os
import random
from typing import Iterable, List, Optional, Tuple, Union

import datasets
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RetrievalDataset(Dataset):
    def __init__(self, args):
        self.args = args
        if os.path.isdir(args.data_dir):
            train_datasets = []
            for file in os.listdir(args.data_dir):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, file),
                    split="train",
                    cache_dir=args.cache_dir_data,
                )

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset("json", data_files=args.train_data, split="train")

        # self.tokenizer = tokenizer
        logger.info("Loaded {} retrieval data.".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Union[dict[str, str], List[BatchEncoding]]:
        query = self.dataset[item]["query"]
        if isinstance(self.dataset[item]["pos"], Iterable):
            pos = random.choice(self.dataset[item]["pos"])
        else:
            pos = self.dataset[item]["pos"]
        if isinstance(self.dataset[item]["neg"], Iterable):
            neg = random.choice(self.dataset[item]["neg"])
        else:
            neg = self.dataset[item]["neg"][0]
        sample = {"query": query, "pos": pos, "neg": neg}
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
        negetive_numbers: Optional[int] = None,
    ):
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.negative_numbers = negetive_numbers

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

        if positive_key and positive_key in dataset:
            dataset = self.generate_samples(dataset)

        self.dataset = dataset
        logger.info("Loaded {} rerank data.".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
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
                negative_samples = negative_samples[: self.negative_numbers]
            for text_neg in negative_samples:
                samples.append([data[self.query_key], text_neg, 0])
        return samples

    @classmethod
    def from_local(cls):
        return
