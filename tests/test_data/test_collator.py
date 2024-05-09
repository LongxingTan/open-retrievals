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
            {'query': 'how are you', 'positive': 'fine'},
            {'query': 'hallo?', 'positive': 'what is your problem'},
        ]

        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = PairCollator(tokenizer=tokenizer)
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 5]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 5]))
        self.assertEqual(batch['pos']['input_ids'].shape, torch.Size([2, 6]))
        self.assertEqual(batch['pos']['attention_mask'].shape, torch.Size([2, 6]))

    def test_triplet_collator(self):
        features = [
            {'query': 'how are you', 'positive': 'fine', 'negative': 'and you?'},
            {'query': 'hallo?', 'positive': 'what is your problem', 'negative': 'I am a doctor'},
        ]

        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = TripletCollator(tokenizer=tokenizer)
        batch = data_collator(features)
        self.assertEqual(batch['query']['input_ids'].shape, torch.Size([2, 5]))
        self.assertEqual(batch['query']['attention_mask'].shape, torch.Size([2, 5]))
        self.assertEqual(batch['pos']['input_ids'].shape, torch.Size([2, 6]))
        self.assertEqual(batch['pos']['attention_mask'].shape, torch.Size([2, 6]))
        self.assertEqual(batch['neg']['input_ids'].shape, torch.Size([2, 6]))
        self.assertEqual(batch['neg']['attention_mask'].shape, torch.Size([2, 6]))
