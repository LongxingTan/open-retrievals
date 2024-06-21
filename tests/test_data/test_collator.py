import os
import shutil
import tempfile
from unittest import TestCase

import torch
from transformers import BertTokenizer

from src.retrievals.data.collator import (
    ColBertCollator,
    LLMRerankCollator,
    PairCollator,
    RerankCollator,
    TripletCollator,
)


class CollatorTest(TestCase):
    def setUp(self) -> None:
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        self.tokenizer = BertTokenizer(self.vocab_file)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_pair_collator(self):
        features = [
            {'query': 'how are you', 'document': 'fine'},
            {'query': 'hallo?', 'document': 'what is your problem'},
        ]

        data_collator = PairCollator(
            tokenizer=self.tokenizer, query_max_length=10, document_max_length=11, document_key='document'
        )
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['document']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['document']['attention_mask'].shape, torch.Size([2, 11]))

    def test_triplet_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine', 'negative': 'and you?'},
            {'query': 'hallo?', 'positive': 'what is your problem', 'negative': 'I am a doctor'},
        ]

        data_collator = TripletCollator(tokenizer=self.tokenizer, query_max_length=10, document_max_length=11)
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['positive']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['positive']['attention_mask'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['negative']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['negative']['attention_mask'].shape, torch.Size([2, 11]))

    def test_rerank_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine'},
            {'query': 'hallo?', 'positive': 'what is your problem'},
        ]

        data_collator = RerankCollator(tokenizer=self.tokenizer, max_length=11, document_key='positive')
        batch = data_collator(features)
        self.assertEqual(batch['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['attention_mask'].shape, torch.Size([2, 11]))

    def test_llm_rerank_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine'},
            {'query': 'hallo?', 'positive': 'what is your problem'},
        ]

        data_collator = LLMRerankCollator(tokenizer=self.tokenizer, max_length=11)
        batch = data_collator(features)
        self.assertEqual(batch['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['attention_mask'].shape, torch.Size([2, 11]))

    def test_colbert_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine', 'negative': 'and you?'},
            {'query': 'hallo?', 'positive': 'what is your problem', 'negative': 'I am a doctor'},
        ]
        data_collator = ColBertCollator(
            tokenizer=self.tokenizer, query_max_length=9, document_max_length=11, positive_key='positive'
        )
        batch = data_collator(features)
        self.assertEqual(batch['query_input_ids'].shape, torch.Size([2, 9]))
        self.assertEqual(batch['query_attention_mask'].shape, torch.Size([2, 9]))
        self.assertEqual(batch['pos_input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['pos_attention_mask'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['neg_input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['neg_attention_mask'].shape, torch.Size([2, 11]))
