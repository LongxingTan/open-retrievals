import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer


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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
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


class RerankDataset(Dataset):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        query = self.dataset[item]["query"]
        passage = self.dataset[item]['passage']
        labels = self.dataset[item]['labels']
        sample = {"query": query, "passage": passage, "neg": labels}
        return sample
