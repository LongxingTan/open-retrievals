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


class RetrievalCollator(DataCollatorWithPadding):
    """Choose the collator based on data/task, support both pair and triplet"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_lengths: List[int], keys: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.keys = keys
        self.max_lengths = max_lengths

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) > 0, "Collator input should not be empty"
        if self.keys is None:
            if isinstance(features[0], dict):
                self.keys = features[0].keys()
            else:
                raise TypeError('Please provide the dict type for each example')

        if len(self.keys) != len(self.max_lengths):
            raise ValueError(
                f'Length of keys and max_lengths should be same, while get {len(self.keys)} and {len(self.max_lengths)}'
            )

        texts = {key: [] for key in self.keys}
        for feature in features:
            for key in self.keys:
                if key in feature:
                    texts[key].append(feature[key])

        tokenize_fn = self.tokenizer
        result = {}
        for i, key in enumerate(self.keys):
            result[key] = self._flatten_and_tokenize(texts[key], self.max_lengths[i], tokenize_fn)

        return result

    def _flatten_and_tokenize(self, texts: List[Any], max_length: int, tokenize_fn) -> Dict[str, Any]:
        if isinstance(texts[0], list):
            texts = sum(texts, [])  # Flatten nested lists

        tokenize_args = {
            "padding": "max_length",
            "max_length": max_length,
            "return_tensors": "pt",
            "truncation": True,
        }
        return tokenize_fn(texts, **tokenize_args)


class RerankCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        query_key: str = 'query',
        document_key: str = 'document',
    ):
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.max_length = max_length
        self.query_key = query_key
        self.document_key = document_key

    def __call__(self, features: Union[List[Dict[str, Any]], List]) -> BatchEncoding:
        assert len(features) > 0, "Make sure the collator is not empty"

        if isinstance(features[0], list):
            features = sum(features, [])

        if isinstance(features[0], dict):
            assert (
                self.query_key in features[0] and self.document_key in features[0]
            ), f"RerankCollator should have '{self.query_key}' and '{self.document_key}' keys in features, maybe labels"
            query_texts = [feature[self.query_key] for feature in features]
            document_texts = [feature[self.document_key] for feature in features]
        else:
            query_texts = [feature[0] for feature in features]
            document_texts = [feature[1] for feature in features]

        if isinstance(query_texts[0], str):
            tokenize_fn = self.tokenizer
            tokenize_args = {
                "truncation": True,
            }
        else:
            tokenize_fn = self.tokenizer.pad
            tokenize_args = {
                "pad_to_multiple_of": None,
            }

        batch = tokenize_fn(
            text=query_texts,
            text_pair=document_texts,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            **tokenize_args,
        )

        if 'labels' in features[0].keys():
            labels = [feature['labels'] for feature in features]
            batch['labels'] = torch.tensor(labels, dtype=torch.float32)
        return batch


class ColBertCollator(DataCollatorWithPadding):
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
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) > 0
        assert (
            self.query_key in features[0] and self.positive_key in features[0]
        ), f"Collator should have '{self.query_key}' and '{self.positive_key}' in features, while get {features[0]}"

        query_texts = [feature[self.query_key] for feature in features]
        pos_texts = [feature[self.positive_key] for feature in features]

        if isinstance(query_texts[0], list):
            query_texts = sum(query_texts, [])
        if isinstance(pos_texts[0], list):
            pos_texts = sum(pos_texts, [])

        if isinstance(query_texts[0], str):
            tokenize_fn = self.tokenizer
            tokenize_args = {
                "truncation": True,
            }
        else:
            tokenize_fn = self.tokenizer.pad
            tokenize_args = {
                "pad_to_multiple_of": None,
            }

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
                neg_texts = sum(neg_texts, [])  # flatten nested list

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
    """Rerank collator for causal llm, with examples query, positive and negative"""

    tokenizers: PreTrainedTokenizer
    query_key: str = 'query'
    positive_key: str = 'positive'
    negative_key: str = 'negative'
    max_length: int = 128

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        add_target_token: str = '',
        sep_token: str = "\n",
        max_length: int = 128,
        tokenize_args: Optional[Dict] = None,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.add_target_token = add_target_token
        self.sep_token = sep_token
        self.bos_token = tokenizer.bos_token if tokenizer.bos_token else ''
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]], return_tensors='pt'):
        examples = []

        if isinstance(features[0], dict):
            """explode the {(query, positive, negatives)} to pair data"""
            for i in range(len(features)):
                examples.append((features[i][self.query_key], features[i][self.positive_key]))
                for neg in features[i][self.negative_key]:
                    examples.append((features[i][self.query_key], neg))
        else:
            examples = features

        # TODO: double check the add_target_token, only yes now?
        batch = self.tokenizer(
            [self.bos_token + i[0] for i in examples],
            [self.sep_token + i[1] + self.sep_token + self.prompt + self.add_target_token for i in examples],
            return_tensors=None,
            max_length=self.max_length,
            truncation='only_second',
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        batch['attention_mask'] = [[1] * len(example) for example in batch['input_ids']]

        if self.add_target_token:
            batch['labels'] = batch['input_ids'].copy()
            batch['labels'] = [[-100] * (len(example) - 1) + example[-1:] for example in batch['labels']]

            max_label_length = max(len(l) for l in batch['labels'])
            padding_side = self.tokenizer.padding_side
            for i in range(len(batch['labels'])):
                feature = batch['labels'][i]
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature))
                if isinstance(feature, list):
                    batch['labels'][i] = feature + remainder if padding_side == "right" else remainder + feature
                elif padding_side == "right":
                    batch['labels'][i] = np.concatenate([feature, remainder]).astype(np.int64)
                else:
                    batch['labels'][i] = np.concatenate([remainder, feature]).astype(np.int64)

        batch = self.tokenizer.pad(
            batch,
            padding='longest',
            max_length=self.max_length,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return batch


class EncodeCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer: PreTrainedTokenizer, id_key: Optional[str] = None, **kwargs):
        self.tokenizer = tokenizer
        self.id_key = id_key

    def __call__(self, features):
        if self.id_key is not None:
            text_ids = [x[0] for x in features]
            text_features = [x[1] for x in features]
            collated_features = super().__call__(text_features)
            return text_ids, collated_features
        else:
            text_features = features
            collated_features = super().__call__(text_features)
            return collated_features


def mask_pad_token(q: Dict[str, torch.Tensor], prob=0.9):
    if random.random() > prob:
        tensor = q['input_ids'].float()
        mask = torch.rand(tensor.shape)
        mask = (mask > prob).float()
        tensor = tensor * (1 - mask) + 2 * mask
        tensor = tensor.long()
        q['input_ids'] = tensor
    return q
