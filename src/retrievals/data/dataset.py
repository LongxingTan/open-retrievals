import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing.

    Args:
        data_name_or_path: Path to dataset or dataset name
        train_group_size: Number of samples per training group
        unfold_each_positive: Whether to unfold each positive example
        query_key: Key for query text in dataset
        positive_key: Key for positive document text in dataset
        negative_key: Key for negative document text in dataset
        query_instruction: Template for query text
        document_instruction: Template for document text
        separator: Separator for title and text
        dataset_split: Dataset split to use
        dataset_language: Dataset language
    """

    data_name_or_path: Union[str, datasets.Dataset, None]
    train_group_size: int = 2
    unfold_each_positive: bool = False
    query_key: str = 'query'
    positive_key: str = 'positive'
    negative_key: str = 'negative'
    query_instruction: str = '{}'
    document_instruction: str = '{}'
    separator: str = ' '
    dataset_split: str = 'train'
    dataset_language: str = 'default'


class RetrievalTrainDataset(Dataset):
    """Dataset for retrieval training that supports query-positive-negative triplets.

    This dataset handles loading and processing of retrieval training data,
    supporting both single and multiple positive/negative examples per query.

    Args:
        config: Dataset configuration
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        if not config and not kwargs.get('data_name_or_path'):
            raise ValueError('Provide either config or data_name_or_path')

        if config:
            self.config = config
        else:
            self.config = DatasetConfig(**kwargs)

        self.tokenizer = tokenizer
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} retrieval examples")

    def _load_dataset(self) -> datasets.Dataset:
        """Load and process the dataset.

        Returns:
            datasets.Dataset: Processed dataset

        Raises:
            ValueError: If dataset loading fails
        """
        if isinstance(self.config.data_name_or_path, datasets.Dataset):
            dataset = self.config.data_name_or_path
        elif os.path.isdir(self.config.data_name_or_path):
            train_datasets = []
            for file in os.listdir(self.config.data_name_or_path):
                if not file.endswith(('.json', '.jsonl')):
                    continue
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(self.config.data_name_or_path, file),
                )
                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)
        else:
            if self.config.data_name_or_path.endswith(('.jsonl', '.json')):
                dataset = datasets.load_dataset("json", data_files=self.config.data_name_or_path)
            else:
                dataset = datasets.load_dataset(self.config.data_name_or_path, self.config.dataset_language)

        if self.config.dataset_split in dataset:
            dataset = dataset[self.config.dataset_split]

        # Filter out examples without positive/negative samples
        dataset = dataset.filter(lambda x: len(x[self.config.positive_key]) > 0)
        if self.config.negative_key in dataset[0]:
            dataset = dataset.filter(lambda x: len(x[self.config.negative_key]) > 0)

        if self.config.unfold_each_positive:
            return self._generate_unfold_samples(dataset)
        return dataset

    def _generate_unfold_samples(self, dataset: datasets.Dataset) -> List[Dict[str, Any]]:
        """Generate samples with unfolded positive examples.

        Args:
            dataset: Input dataset

        Returns:
            List[Dict[str, Any]]: List of processed samples
        """
        samples = []
        for data in dataset:
            for pos_text in data[self.config.positive_key]:
                sample = {
                    self.config.query_key: self.config.query_instruction.format(data[self.config.query_key]),
                    self.config.positive_key: self.config.document_instruction.format(pos_text),
                }

                if self.config.negative_key in data:
                    negs = self._process_negative_samples(data[self.config.negative_key])
                    sample[self.config.negative_key] = [self.config.document_instruction.format(neg) for neg in negs]
                samples.append(sample)
        return samples

    def _process_negative_samples(self, negatives: Union[List[str], str]) -> List[str]:
        """Process negative samples to get the required number.

        Args:
            negatives: List of negative samples or single negative sample

        Returns:
            List[str]: Processed negative samples
        """
        if isinstance(negatives, (list, tuple)):
            if len(negatives) < self.config.train_group_size - 1:
                num = math.ceil((self.config.train_group_size - 1) / len(negatives))
                return random.sample(negatives * num, self.config.train_group_size - 1)
            return random.sample(negatives, self.config.train_group_size - 1)
        return [negatives]

    def _format_document(self, doc: Union[str, Dict[str, str]]) -> str:
        """Format document text with title if available.

        Args:
            doc: Document text or dictionary with title and text

        Returns:
            str: Formatted document text
        """
        if isinstance(doc, dict):
            text = doc['title'] + self.config.separator + doc['text'] if 'title' in doc else doc['text']
        else:
            text = doc
        return self.config.document_instruction.format(text)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a training example.

        Args:
            item: Index of the example

        Returns:
            Dict[str, Any]: Processed example
        """
        if self.config.unfold_each_positive:
            return self.dataset[item]

        data = self.dataset[item]
        query = self.config.query_instruction.format(data[self.config.query_key])

        # Process positive example
        if isinstance(data[self.config.positive_key], (list, tuple)):
            pos = random.choice(data[self.config.positive_key])
            pos = self._format_document(pos)
        else:
            pos = self._format_document(data[self.config.positive_key])

        sample = {self.config.query_key: query, self.config.positive_key: pos}

        # Process negative examples
        if self.config.negative_key in data:
            negs = self._process_negative_samples(data[self.config.negative_key])
            sample[self.config.negative_key] = [self._format_document(neg) for neg in negs]

        return sample


class RerankTrainDataset(Dataset):
    """Dataset for reranking training that supports query-document pairs.

    This dataset handles loading and processing of reranking training data,
    supporting both single and multiple positive/negative examples per query.

    Args:
        config: Dataset configuration
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        if not config and not kwargs.get('data_name_or_path'):
            raise ValueError('Provide either config or data_name_or_path')

        if config:
            self.config = config
        else:
            self.config = DatasetConfig(**kwargs)

        self.tokenizer = tokenizer
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} reranking examples")

    def _load_dataset(self) -> datasets.Dataset:
        """Load and process the dataset.

        Returns:
            datasets.Dataset: Processed dataset

        Raises:
            ValueError: If dataset loading fails
        """
        if isinstance(self.config.data_name_or_path, datasets.Dataset):
            dataset = self.config.data_name_or_path
        elif os.path.isdir(self.config.data_name_or_path):
            train_datasets = []
            for file in os.listdir(self.config.data_name_or_path):
                if not file.endswith(('.json', '.jsonl')):
                    continue
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(self.config.data_name_or_path, file),
                )
                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)
        else:
            if self.config.data_name_or_path.endswith(('.jsonl', '.json')):
                dataset = datasets.load_dataset("json", data_files=self.config.data_name_or_path)
            else:
                dataset = datasets.load_dataset(self.config.data_name_or_path)

        if self.config.dataset_split in dataset:
            dataset = dataset[self.config.dataset_split]

        # Filter out examples without positive/negative samples
        dataset = dataset.filter(lambda x: len(x[self.config.positive_key]) > 0)
        if self.config.negative_key in dataset[0]:
            dataset = dataset.filter(lambda x: len(x[self.config.negative_key]) > 0)

        return self._generate_samples(dataset)

    def _generate_samples(self, dataset: datasets.Dataset) -> List[Dict[str, Any]]:
        """Generate training samples from the dataset.

        Args:
            dataset: Input dataset

        Returns:
            List[Dict[str, Any]]: List of processed samples
        """
        samples = []
        for data in dataset:
            query = self.config.query_instruction.format(data[self.config.query_key])

            # Process positive example
            if isinstance(data[self.config.positive_key], (list, tuple)):
                pos = random.choice(data[self.config.positive_key])
                pos = self._format_document(pos)
            else:
                pos = self._format_document(data[self.config.positive_key])

            sample = {self.config.query_key: query, self.config.positive_key: pos}

            # Process negative examples
            if self.config.negative_key in data:
                negs = self._process_negative_samples(data[self.config.negative_key])
                sample[self.config.negative_key] = [self._format_document(neg) for neg in negs]

            samples.append(sample)
        return samples

    def _process_negative_samples(self, negatives: Union[List[str], str]) -> List[str]:
        """Process negative samples to get the required number.

        Args:
            negatives: List of negative samples or single negative sample

        Returns:
            List[str]: Processed negative samples
        """
        if isinstance(negatives, (list, tuple)):
            if len(negatives) < self.config.train_group_size - 1:
                num = math.ceil((self.config.train_group_size - 1) / len(negatives))
                return random.sample(negatives * num, self.config.train_group_size - 1)
            return random.sample(negatives, self.config.train_group_size - 1)
        return [negatives]

    def _format_document(self, doc: Union[str, Dict[str, str]]) -> str:
        """Format document text with title if available.

        Args:
            doc: Document text or dictionary with title and text

        Returns:
            str: Formatted document text
        """
        if isinstance(doc, dict):
            text = doc['title'] + self.config.separator + doc['text'] if 'title' in doc else doc['text']
        else:
            text = doc
        return self.config.document_instruction.format(text)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a training example.

        Args:
            item: Index of the example

        Returns:
            Dict[str, Any]: Processed example
        """
        return self.dataset[item]


class EncodeDataset(Dataset):
    """Dataset for encoding tasks that processes text inputs with optional IDs.

    This dataset handles loading and processing of text data for encoding,
    supporting optional ID tracking for each input.

    Args:
        config: Dataset configuration
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        if not config and not kwargs.get('data_name_or_path'):
            raise ValueError('Provide either config or data_name_or_path')

        if config:
            self.config = config
        else:
            self.config = DatasetConfig(**kwargs)

        self.tokenizer = tokenizer
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} encoding examples")

    def _load_dataset(self) -> datasets.Dataset:
        """Load and process the dataset.

        Returns:
            datasets.Dataset: Processed dataset

        Raises:
            ValueError: If dataset loading fails
        """
        if isinstance(self.config.data_name_or_path, datasets.Dataset):
            dataset = self.config.data_name_or_path
        elif os.path.isdir(self.config.data_name_or_path):
            train_datasets = []
            for file in os.listdir(self.config.data_name_or_path):
                if not file.endswith(('.json', '.jsonl')):
                    continue
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(self.config.data_name_or_path, file),
                )
                train_datasets.append(temp_dataset)
            dataset = datasets.concatenate_datasets(train_datasets)
        else:
            if self.config.data_name_or_path.endswith(('.jsonl', '.json')):
                dataset = datasets.load_dataset("json", data_files=self.config.data_name_or_path)
            else:
                dataset = datasets.load_dataset(self.config.data_name_or_path, self.config.dataset_language)

        if self.config.dataset_split in dataset:
            dataset = dataset[self.config.dataset_split]

        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[Optional[str], Dict[str, Any]]:
        """Get an encoding example.

        Args:
            item: Index of the example

        Returns:
            Tuple[Optional[str], Dict[str, Any]]: ID and processed example
        """
        data = self.dataset[item]
        text = self.config.query_instruction.format(data[self.config.query_key])

        if self.config.id_key and self.config.id_key in data:
            return data[self.config.id_key], {'text': text}
        return None, {'text': text}


class RerankDataset(Dataset):
    """Rerank inference dataset"""

    def __init__(self):
        pass
