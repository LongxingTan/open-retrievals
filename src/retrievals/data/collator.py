from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer


class PairCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        passage_max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_max_length: int
        self.passage_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.passage_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.passage_max_length = tokenizer.model_max_length

        if passage_max_length:
            self.passage_max_length = passage_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert (
            'query' in features[0] and 'pos' in features[0]
        ), "PairCollator should have 'query' and 'pos' keys in features dict"

        query_texts = [feature["query"] for feature in features]
        pos_texts = [feature["pos"] for feature in features]

        query_inputs = self.tokenizer(
            query_texts,
            padding=True,
            max_length=self.query_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pos_inputs = self.tokenizer(
            pos_texts,
            padding=True,
            max_length=self.passage_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {"query": query_inputs, "pos": pos_inputs}


class TripletCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        passage_max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_max_length: int
        self.passage_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.passage_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.passage_max_length = tokenizer.model_max_length

        if passage_max_length:
            self.passage_max_length = passage_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert (
            'query' in features[0] and 'pos' in features[0] and 'neg' in features[0]
        ), "TripletCollator should have 'query', 'pos' and 'neg' keys in features dict"

        query_texts = [feature["query"] for feature in features]
        pos_texts = [feature["pos"] for feature in features]
        neg_texts = [feature["neg"] for feature in features]

        # if isinstance(query[0], list):
        #     query = sum(query, [])
        # if isinstance(passage[0], list):
        #     passage = sum(passage, [])

        query_inputs = self.tokenizer(
            query_texts,
            padding=True,
            max_length=self.query_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pos_inputs = self.tokenizer(
            pos_texts,
            padding=True,
            max_length=self.passage_max_length,
            truncation=True,
            return_tensors="pt",
        )  # ["input_ids"]
        neg_inputs = self.tokenizer(
            neg_texts,
            padding=True,
            max_length=self.passage_max_length,
            truncation=True,
            return_tensors="pt",
        )  # ["input_ids"]

        return {
            "query": query_inputs,
            "pos": pos_inputs,
            "neg": neg_inputs,
        }


class RerankCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        passage_max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_max_length: int
        self.passage_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.passage_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.passage_max_length = tokenizer.model_max_length

        if passage_max_length:
            self.passage_max_length = passage_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        assert (
            'query' in features and 'passage' in features
        ), "RerankCollator should have 'query' and 'passage' keys in features dict, and 'labels' during training"

        query_texts = [feature["query"] for feature in features]
        passage_texts = [feature['passage'] for feature in features]

        labels = None
        if 'labels' in features[0].keys():
            labels = [feature['labels'] for feature in features]

        batch = self.tokenizer(
            text=query_texts, text_pair=passage_texts, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # for key in ['input_ids', 'attention_mask']:
        #     batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.float32)
        return batch
