import os
import shutil
import tempfile
from unittest import TestCase

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer

from src.retrievals.data.collator import RerankCollator
from src.retrievals.models.rerank import AutoModelForRanking, ColBERT

from .test_modeling_common import (
    ModelTesterMixin,
    device,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)

# class AutoModelForRankingTest(TestCase, ModelTesterMixin):
#     def setUp(self) -> None:
#         self.output_dir = tempfile.mkdtemp()
#         # self.config_tester = ConfigTester()
#         model_name_or_path = 'BAAI/bge-reranker-base'
#         self.data_collator = RerankCollator(AutoTokenizer.from_pretrained(model_name_or_path))
#         self.model = AutoModelForRanking.from_pretrained(model_name_or_path, temperature=0.05)
#
#         vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
#         self.vocab_file = os.path.join(self.output_dir, "vocab.txt")
#         with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
#             vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
#
#         self.tokenizer = BertTokenizer(self.vocab_file)
#
#     def tearDown(self):
#         shutil.rmtree(self.output_dir)
#
#     def test_preprocess(self):
#         text = '张华考上了北京大学'
#         text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
#         text_pairs = [[text, i] for i in text_list]
#
#         batch = self.model.preprocess_pair(text_pairs, max_length=9)
#         self.assertEqual(batch['input_ids'].shape, torch.Size([3, 9]))
#         self.assertEqual(batch['attention_mask'].shape, torch.Size([3, 9]))
#
#     def test_compute_score(self):
#         text = '张华考上了北京大学'
#         text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
#         text_pairs = [[text, i] for i in text_list]
#         scores = self.model.compute_score(sentence_pairs=text_pairs, data_collator=self.data_collator)
#         document_ranked = self.model.rerank(query=text, documents=text_list, data_collator=self.data_collator)
#
#         print(scores)
#         print(document_ranked)


class TestColBERT(TestCase):
    def setUp(self):
        # Create mock objects
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.Linear(768, 256)
        self.colbert = ColBERT(
            model=self.model, tokenizer=self.tokenizer, linear_layer=self.linear_layer, max_length=128
        )

    def test_forward(self):
        query_input_ids = torch.tensor([[101, 2054, 2003, 1996, 1045, 2572, 102]])
        query_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        pos_input_ids = torch.tensor([[101, 2054, 2003, 1996, 1045, 2572, 102]])
        pos_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

        self.colbert.train()  # Set the model to training mode
        output = self.colbert(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
        )
        self.assertIn('loss', output)
        self.assertIsInstance(output['loss'], torch.Tensor)

    def test_preprocess_pair(self):
        batch_sentence_pair = [["Hello, world!", "Hello there!"]]
        preprocessed = self.colbert.preprocess_pair(
            batch_sentence_pair=batch_sentence_pair, query_max_length=128, document_max_length=128
        )
        self.assertIn("query_input_ids", preprocessed)
        self.assertIn("doc_input_ids", preprocessed)
        self.assertIsInstance(preprocessed["query_input_ids"], torch.Tensor)
        self.assertIsInstance(preprocessed["doc_input_ids"], torch.Tensor)
