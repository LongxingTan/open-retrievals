from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer


class PairCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_key: str = 'query',
        positive_key: str = 'positive',
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        document_max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_key = query_key
        self.positive_key = positive_key

        self.query_max_length: int
        self.document_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.document_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.document_max_length = tokenizer.model_max_length

        if document_max_length:
            self.document_max_length = document_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) > 0
        assert (
            self.query_key in features[0] and self.positive_key in features[0]
        ), f"PairCollator should have {self.query_key} and {self.positive_key} in features, while get {features[0]}"
        "you can set the custom key of query_key, positive_key during class init"

        query_texts = [feature["query"] for feature in features]
        pos_texts = [feature["positive"] for feature in features]

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
            max_length=self.document_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {"query": query_inputs, "positive": pos_inputs}


class TripletCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_key: str = 'query',
        positive_key: str = 'positive',
        negative_key: Optional[str] = 'negative',
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        document_max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key

        self.query_max_length: int
        self.document_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.document_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.document_max_length = tokenizer.model_max_length

        if document_max_length:
            self.document_max_length = document_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) > 0
        assert (
            self.positive_key in features[0] and self.positive_key in features[0] and self.negative_key in features[0]
        ), f"TripletCollator should have {self.query_key}, {self.positive_key} and {self.negative_key} in features dict"
        "you can set the custom key of query_key, positive_key and negative_key during class init"

        query_texts = [feature[self.query_key] for feature in features]
        pos_texts = [feature[self.positive_key] for feature in features]
        neg_texts = [feature[self.negative_key] for feature in features]

        # if isinstance(query[0], list):
        #     query = sum(query, [])
        # if isinstance(document[0], list):
        #     document = sum(document, [])

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
            max_length=self.document_max_length,
            truncation=True,
            return_tensors="pt",
        )  # ["input_ids"]
        neg_inputs = self.tokenizer(
            neg_texts,
            padding=True,
            max_length=self.document_max_length,
            truncation=True,
            return_tensors="pt",
        )  # ["input_ids"]

        return {
            "query": query_inputs,
            "positive": pos_inputs,
            "negative": neg_inputs,
        }


class RerankCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_key: str = 'query',
        document_key: str = 'document',
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        document_max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_key = query_key
        self.document_key = document_key

        self.query_max_length: int
        self.document_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        if document_max_length:
            self.document_max_length = document_max_length
        elif max_length:
            self.query_max_length = max_length
            self.document_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.document_max_length = tokenizer.model_max_length

    def __call__(self, features: Union[List[Dict[str, Any]], List]) -> BatchEncoding:
        assert len(features) > 0
        if isinstance(features[0], dict):
            assert (
                self.query_key in features[0] and self.document_key in features[0]
            ), f"RerankCollator should have {self.query_key} and {self.document_key} keys in features, "
            "and 'labels' during training"
            query_texts = [feature[self.query_key] for feature in features]
            document_texts = [feature[self.document_key] for feature in features]
        else:
            query_texts = [feature[0] for feature in features]
            document_texts = [feature[1] for feature in features]

        batch = self.tokenizer(
            text=query_texts,
            text_pair=document_texts,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )

        if 'labels' in features[0].keys():
            labels = [feature['labels'] for feature in features]
            batch['labels'] = torch.tensor(labels, dtype=torch.float32)
        return batch


class ColBertCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_key: str = 'query',
        positive_key: str = 'positive',
        negative_key: str = 'negative',
        max_length: Optional[int] = None,
        query_max_length: Optional[int] = None,
        document_max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.query_key = query_key
        self.positive_key = positive_key
        self.negative_key = negative_key

        self.query_max_length: int
        self.document_max_length: int
        if query_max_length:
            self.query_max_length = query_max_length
        elif max_length:
            self.query_max_length = max_length
            self.document_max_length = max_length
        else:
            self.query_max_length = tokenizer.model_max_length
            self.document_max_length = tokenizer.model_max_length

        if document_max_length:
            self.document_max_length = document_max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) > 0
        assert (
            self.query_key in features[0] and self.positive_key in features[0]
        ), f"PairCollator should have {self.query_key} and {self.positive_key} in features, while get {features[0]}"
        "you can set the custom key of query_key, positive_key during class init"

        query_texts = [feature[self.query_key] for feature in features]
        pos_texts = [feature[self.positive_key] for feature in features]

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
            max_length=self.document_max_length,
            truncation=True,
            return_tensors="pt",
        )

        batch = {
            'query_input_ids': query_inputs['input_ids'],
            'query_attention_mask': query_inputs['attention_mask'],
            'pos_input_ids': pos_inputs['input_ids'],
            'pos_attention_mask': pos_inputs['attention_mask'],
        }
        if 'labels' in features[0].keys():
            labels = [feature['labels'] for feature in features]
            batch['labels'] = torch.tensor(labels, dtype=torch.float32)
        return batch
