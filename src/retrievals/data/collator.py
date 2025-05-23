import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    BatchEncoding,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class RetrievalCollator(DataCollatorWithPadding):
    """A collator for retrieval tasks that supports both pair and triplet data formats.

    This collator handles batching and tokenization of retrieval data, supporting multiple
    input fields with different maximum lengths.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text
        max_lengths (List[int]): Maximum sequence lengths for each input field
        keys (Optional[List[str]]): Names of the input fields to process. If None, use all keys from the first example
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_lengths: List[int], keys: Optional[List[str]] = None):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            logger.info("Adding [PAD] token to tokenizer")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.keys = keys
        self.max_lengths = max_lengths

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of features into a format suitable for model input.

        Args:
            features (List[Dict[str, Any]]): List of feature dictionaries to process

        Returns:
            Dict[str, Any]: Processed batch with tokenized inputs

        Raises:
            ValueError: If features list is empty or keys/max_lengths mismatch
            TypeError: If features are not in dictionary format
        """
        if not features:
            raise ValueError("Collator input should not be empty")

        if self.keys is None:
            if isinstance(features[0], dict):
                self.keys = list(features[0].keys())
            else:
                raise TypeError('Features must be provided as dictionaries')

        if len(self.keys) != len(self.max_lengths):
            raise ValueError(f'Length of keys ({len(self.keys)}) and max_lengths ({len(self.max_lengths)}) must match')

        texts_dict = {key: [] for key in self.keys}
        for feature in features:
            for key in self.keys:
                if key in feature:
                    texts_dict[key].append(feature[key])

        result = {}
        for i, key in enumerate(self.keys):
            result[key] = self._flatten_and_tokenize(texts_dict, key, self.max_lengths[i])

        return result

    def _flatten_and_tokenize(self, texts_dict: Dict[str, List[Any]], key: str, max_length: int) -> Dict[str, Any]:
        """Flatten and tokenize a list of texts for a given key.

        Args:
            texts_dict (Dict[str, List[Any]]): Dictionary containing lists of texts
            key (str): The key to process
            max_length (int): Maximum sequence length

        Returns:
            Dict[str, Any]: Tokenized inputs

        Raises:
            ValueError: If the input list is empty
        """
        texts = texts_dict[key]
        if not texts:
            raise ValueError(f"No texts found for key '{key}'")

        if isinstance(texts[0], list):
            texts = sum(texts, [])  # Flatten nested lists

        tokenize_args = {
            "padding": "max_length",
            "max_length": max_length,
            "return_tensors": "pt",
            "truncation": True,
        }
        return self.tokenizer(texts, **tokenize_args)

    def _mask_pad_token(self, q: Dict[str, torch.Tensor], mask_prob: float = 0.9) -> Dict[str, torch.Tensor]:
        """Randomly mask pad tokens in the input.

        Args:
            q (Dict[str, torch.Tensor]): Input tensor dictionary
            mask_prob (float): Probability of masking a token

        Returns:
            Dict[str, torch.Tensor]: Input with masked tokens
        """
        if random.random() > mask_prob:
            tensor = q['input_ids'].float()
            mask = torch.rand(tensor.shape)
            mask = (mask > mask_prob).float()
            tensor = tensor * (1 - mask) + 2 * mask
            tensor = tensor.long()
            q['input_ids'] = tensor
        return q


class RerankCollator(DataCollatorWithPadding):
    """A collator for reranking tasks that processes query-document pairs.

    This collator handles batching and tokenization of query-document pairs for reranking,
    with support for optional labels.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text
        max_length (int): Maximum sequence length for combined query-document input
        query_key (str): Key for query text in input features
        document_key (str): Key for document text in input features
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        query_key: str = 'query',
        document_key: str = 'document',
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            logger.info("Adding [PAD] token to tokenizer")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.max_length = max_length
        self.query_key = query_key
        self.document_key = document_key

    def __call__(self, features: Union[List[Dict[str, Any]], List[Tuple[str, str]]]) -> BatchEncoding:
        """Process a batch of features into a format suitable for reranking.

        Args:
            features: List of feature dictionaries or query-document pairs

        Returns:
            BatchEncoding: Processed batch with tokenized inputs and optional labels

        Raises:
            ValueError: If features list is empty or missing required keys
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        if isinstance(features[0], list):
            features = sum(features, [])

        if isinstance(features[0], dict):
            if not all(key in features[0] for key in [self.query_key, self.document_key]):
                raise ValueError(f"Features must contain '{self.query_key}' and '{self.document_key}' keys")
            query_texts = [feature[self.query_key] for feature in features]
            document_texts = [feature[self.document_key] for feature in features]
        else:
            query_texts = [feature[0] for feature in features]
            document_texts = [feature[1] for feature in features]

        tokenize_fn = self.tokenizer if isinstance(query_texts[0], str) else self.tokenizer.pad
        tokenize_args = {"truncation": True} if isinstance(query_texts[0], str) else {"pad_to_multiple_of": None}

        batch = tokenize_fn(
            text=query_texts,
            text_pair=document_texts,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            **tokenize_args,
        )

        if 'labels' in features[0]:
            labels = [feature['labels'] for feature in features]
            batch['labels'] = torch.tensor(labels, dtype=torch.float32)

        return batch


class ColBertCollator(DataCollatorWithPadding):
    """A collator for ColBERT-style training that processes query-positive-negative triplets.

    This collator handles batching and tokenization for ColBERT training, supporting
    query-positive-negative triplets with separate max lengths for queries and documents.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text
        query_max_length (int): Maximum sequence length for queries
        document_max_length (int): Maximum sequence length for documents
        query_key (str): Key for query text in input features
        positive_key (str): Key for positive document text in input features
        negative_key (str): Key for negative document text in input features
        tokenize_args (Optional[Dict]): Additional arguments for tokenization
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_max_length: int = 32,
        document_max_length: int = 128,
        query_key: str = 'query',
        positive_key: str = 'positive',
        negative_key: str = 'negative',
        tokenize_args: Optional[Dict] = None,
    ) -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            logger.info("Adding [PAD] token to tokenizer")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.tokenize_args = tokenize_args or {}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of features into a format suitable for ColBERT training.

        Args:
            features: List of feature dictionaries containing query-positive-negative triplets

        Returns:
            Dict[str, Any]: Processed batch with tokenized inputs

        Raises:
            ValueError: If features list is empty or missing required keys
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        if not all(key in features[0] for key in [self.query_key, self.positive_key]):
            raise ValueError(f"Features must contain '{self.query_key}' and '{self.positive_key}' keys")

        query_texts = [feature[self.query_key] for feature in features]
        pos_texts = [feature[self.positive_key] for feature in features]

        if isinstance(query_texts[0], list):
            query_texts = sum(query_texts, [])
        if isinstance(pos_texts[0], list):
            pos_texts = sum(pos_texts, [])

        tokenize_fn = self.tokenizer if isinstance(query_texts[0], str) else self.tokenizer.pad
        tokenize_args = {"truncation": True} if isinstance(query_texts[0], str) else {"pad_to_multiple_of": None}
        tokenize_args.update(self.tokenize_args)

        query_inputs = tokenize_fn(
            query_texts, padding="max_length", max_length=self.query_max_length, return_tensors="pt", **tokenize_args
        )
        pos_inputs = tokenize_fn(
            pos_texts, padding="max_length", max_length=self.document_max_length, return_tensors="pt", **tokenize_args
        )

        batch = {
            'query_input_ids': query_inputs['input_ids'],
            'query_attention_mask': query_inputs['attention_mask'],
            'pos_input_ids': pos_inputs['input_ids'],
            'pos_attention_mask': pos_inputs['attention_mask'],
        }

        if self.negative_key in features[0]:
            neg_texts = [feature[self.negative_key] for feature in features]
            if isinstance(neg_texts[0], list):
                neg_texts = sum(neg_texts, [])

            neg_inputs = tokenize_fn(
                neg_texts,
                padding='max_length',
                max_length=self.document_max_length,
                return_tensors='pt',
                **tokenize_args,
            )
            batch.update({'neg_input_ids': neg_inputs['input_ids'], 'neg_attention_mask': neg_inputs['attention_mask']})

        return batch


class LLMRerankCollator(DataCollatorForSeq2Seq):
    """A collator for LLM-based reranking that processes query-positive-negative examples.

    This collator handles batching and tokenization for LLM-based reranking, supporting
    query-positive-negative examples with prompt templates and target tokens.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text
        prompt (str): Template prompt for formatting examples
        add_target_token (str): Additional token to add to the target
        sep_token (str): Token to separate different parts of the input
        max_length (int): Maximum sequence length
        pad_to_multiple_of (Optional[int]): Pad sequence length to multiple of this value
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        add_target_token: str = '',
        sep_token: str = "\n",
        max_length: int = 128,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.add_target_token = add_target_token
        self.sep_token = sep_token
        self.bos_token = tokenizer.bos_token if tokenizer.bos_token else ''
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]], return_tensors: str = 'pt') -> Dict[str, Any]:
        """Process a batch of features into a format suitable for LLM reranking.

        Args:
            features: List of feature dictionaries containing query-positive-negative examples
            return_tensors: Type of tensors to return ('pt' for PyTorch)

        Returns:
            Dict[str, Any]: Processed batch with tokenized inputs

        Raises:
            ValueError: If features list is empty or missing required keys
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        examples = []
        if isinstance(features[0], dict):
            # Convert {(query, positive, negatives)} to pair data
            for feature in features:
                if not all(key in feature for key in ['query', 'positive', 'negative']):
                    raise ValueError("Features must contain 'query', 'positive', and 'negative' keys")
                examples.append((feature['query'], feature['positive']))
                for neg in feature['negative']:
                    examples.append((feature['query'], neg))
        else:
            examples = features

        # Format examples with prompt template
        formatted_examples = []
        for query, doc in examples:
            formatted_text = self.prompt.format(
                query=query, document=doc, sep=self.sep_token, bos=self.bos_token, target=self.add_target_token
            )
            formatted_examples.append(formatted_text)

        # Tokenize formatted examples
        tokenized = self.tokenizer(
            formatted_examples,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return tokenized


class RerankDistillCollator(DataCollatorWithPadding):
    """A collator for reranking distillation that processes student and teacher model inputs.

    This collator handles batching and tokenization for reranking distillation,
    supporting separate max lengths for queries and documents.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer for student model
        teacher_tokenizer (PreTrainedTokenizer): The tokenizer for teacher model
        query_max_length (int): Maximum sequence length for queries
        document_max_length (int): Maximum sequence length for documents
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        teacher_tokenizer: PreTrainedTokenizer,
        query_max_length: int,
        document_max_length: int,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of features for reranking distillation.

        Args:
            features: List of feature dictionaries containing student and teacher inputs

        Returns:
            Dict[str, Any]: Processed batch with tokenized inputs for both student and teacher

        Raises:
            ValueError: If features list is empty or missing required keys
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        # Process student model inputs
        student_query = [f['student_query'] for f in features]
        student_doc = [f['student_document'] for f in features]

        student_query_inputs = self.tokenizer(
            student_query, padding=True, max_length=self.query_max_length, truncation=True, return_tensors="pt"
        )
        student_doc_inputs = self.tokenizer(
            student_doc, padding=True, max_length=self.document_max_length, truncation=True, return_tensors="pt"
        )

        # Process teacher model inputs
        teacher_pairs = [f['teacher_pairs'] for f in features]
        teacher_inputs = self.teacher_tokenizer(
            teacher_pairs, padding=True, max_length=self.document_max_length, truncation=True, return_tensors="pt"
        )

        return {
            'student_query_input_ids': student_query_inputs['input_ids'],
            'student_query_attention_mask': student_query_inputs['attention_mask'],
            'student_doc_input_ids': student_doc_inputs['input_ids'],
            'student_doc_attention_mask': student_doc_inputs['attention_mask'],
            'teacher_input_ids': teacher_inputs['input_ids'],
            'teacher_attention_mask': teacher_inputs['attention_mask'],
        }


class EncodeCollator(DataCollatorWithPadding):
    """A collator for encoding tasks that processes text inputs with optional IDs.

    This collator handles batching and tokenization for encoding tasks,
    supporting optional ID tracking for each input.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text
        id_key (Optional[str]): Key for ID in input features
        **kwargs: Additional arguments for tokenization
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, id_key: Optional[str] = None, **kwargs):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.id_key = id_key
        self.tokenize_kwargs = kwargs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of features for encoding.

        Args:
            features: List of feature dictionaries containing text inputs

        Returns:
            Dict[str, Any]: Processed batch with tokenized inputs and optional IDs

        Raises:
            ValueError: If features list is empty
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        texts = [f['text'] for f in features]
        batch = self.tokenizer(texts, padding=True, return_tensors="pt", **self.tokenize_kwargs)

        if self.id_key and self.id_key in features[0]:
            batch['ids'] = [f[self.id_key] for f in features]

        return batch


def mask_pad_token(q: Dict[str, torch.Tensor], prob: float = 0.9) -> Dict[str, torch.Tensor]:
    """Randomly mask pad tokens in the input tensor.

    Args:
        q (Dict[str, torch.Tensor]): Input tensor dictionary
        prob (float): Probability of masking a token

    Returns:
        Dict[str, torch.Tensor]: Input with masked tokens
    """
    if random.random() > prob:
        tensor = q['input_ids'].float()
        mask = torch.rand(tensor.shape)
        mask = (mask > prob).float()
        tensor = tensor * (1 - mask) + 2 * mask
        tensor = tensor.long()
        q['input_ids'] = tensor
    return q
