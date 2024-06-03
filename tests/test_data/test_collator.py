import os
import shutil
import tempfile
from unittest import TestCase

import torch
from transformers import BertTokenizer

from src.retrievals.data.collator import PairCollator, TripletCollator


class CollatorTest(TestCase):
    def setUp(self) -> None:
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_pair_collator(self):
        features = [
            {'query': 'how are you', 'document': 'fine'},
            {'query': 'hallo?', 'document': 'what is your problem'},
        ]

        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = PairCollator(
            tokenizer=tokenizer, query_max_length=10, document_max_length=11, document_key='document'
        )
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['positive']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['positive']['attention_mask'].shape, torch.Size([2, 11]))

    def test_triplet_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine', 'negative': 'and you?'},
            {'query': 'hallo?', 'positive': 'what is your problem', 'negative': 'I am a doctor'},
        ]

        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = TripletCollator(tokenizer=tokenizer, query_max_length=10, document_max_length=11)
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 10]))
        self.assertEqual(batch['positive']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['positive']['attention_mask'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['negative']['input_ids'].shape, torch.Size([2, 11]))
        self.assertEqual(batch['negative']['attention_mask'].shape, torch.Size([2, 11]))
