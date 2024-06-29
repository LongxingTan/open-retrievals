import os
import shutil
import tempfile
from unittest import TestCase

import torch
from transformers import AutoConfig, AutoTokenizer, BertTokenizer

from src.retrievals.data.collator import RerankCollator
from src.retrievals.models.rerank import AutoModelForRanking, ColBERT

from .test_modeling_common import (
    ModelTesterMixin,
    device,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class AutoModelForRankingTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        # self.config_tester = ConfigTester()
        model_name_or_path = 'BAAI/bge-reranker-base'
        self.data_collator = RerankCollator(AutoTokenizer.from_pretrained(model_name_or_path))
        self.model = AutoModelForRanking.from_pretrained(model_name_or_path, temperature=0.05)

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.output_dir, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        self.tokenizer = BertTokenizer(self.vocab_file)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_preprocess(self):
        text = '张华考上了北京大学'
        text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
        text_pairs = [[text, i] for i in text_list]

        batch = self.model.preprocess(text_pairs, max_length=9)
        self.assertEqual(batch['input_ids'].shape, torch.Size([3, 9]))
        self.assertEqual(batch['attention_mask'].shape, torch.Size([3, 9]))

    def test_compute_score(self):
        text = '张华考上了北京大学'
        text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
        text_pairs = [[text, i] for i in text_list]
        scores = self.model.compute_score(sentence_pairs=text_pairs, data_collator=self.data_collator)
        document_ranked = self.model.rerank(query=text, documents=text_list, data_collator=self.data_collator)

        print(scores)
        print(document_ranked)
