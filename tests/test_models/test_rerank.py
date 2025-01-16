import os
import shutil
import tempfile
from dataclasses import dataclass, field
from unittest import TestCase

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from src.retrievals.data.collator import RerankCollator
from src.retrievals.models.rerank import AutoModelForRanking, ColBERT
from src.retrievals.trainer.trainer import (
    DistilTrainer,
    RerankTrainer,
    RetrievalTrainer,
)

from .test_modeling_common import (
    ModelTesterMixin,
    device,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class PseudoRerankTrainDataset(Dataset):
    def __init__(self):
        self.examples = [
            {'query': 'how are you', 'document': 'fine', 'labels': 1},
            {'query': 'hallo?', 'document': 'what is your problem', 'labels': 1},
            {'query': 'how are you doing', 'document': 'survive', 'labels': 0},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class AutoModelForRankingTest(TestCase, ModelTesterMixin):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = 'BAAI/bge-reranker-base'
        self.data_collator = RerankCollator(AutoTokenizer.from_pretrained(model_name_or_path))
        self.model = AutoModelForRanking.from_pretrained(model_name_or_path, device='cpu', temperature=0.05)

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.output_dir, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        self.tokenizer = BertTokenizer(self.vocab_file)

        self.text = '张华考上了北京大学'
        self.text_list = ['李萍进了中等技术学校', '我在百货公司当售货员', '我们都有光明的前途']
        self.text_pairs = [[self.text, i] for i in self.text_list]

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_preprocess(self):

        batch = self.model.preprocess_pair(self.text_pairs, query_max_length=9, document_max_length=9)
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)

    def test_compute_score(self):

        scores = self.model.compute_score(sentence_pairs=self.text_pairs, data_collator=self.data_collator)
        document_ranked = self.model.rerank(query=self.text, documents=self.text_list, data_collator=self.data_collator)

        self.assertEqual(len(scores), len(self.text_pairs))
        self.assertIn('rerank_document', document_ranked)
        self.assertIn('rerank_scores', document_ranked)

    def test_trainer(self):
        # Use a mock dataset for testing
        train_dataset = PseudoRerankTrainDataset()
        training_args = TrainingArguments(output_dir=self.output_dir, per_device_train_batch_size=2, num_train_epochs=1)

        self.trainer = RerankTrainer(
            model=self.model, args=training_args, train_dataset=train_dataset, loss_fn=nn.BCEWithLogitsLoss()
        )

        inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),  # Batch of size 1, sequence length 3
            'attention_mask': torch.tensor([[1, 1, 1]]),  # Attention mask (all tokens are attended to)
            'labels': torch.tensor([1]),  # Labels for loss computation (assuming binary classification)
        }

        loss, outputs = self.trainer.compute_loss(self.model, inputs, return_outputs=True)
        self.assertIsInstance(loss, torch.Tensor)

        self.assertIn('logits', outputs, "Model outputs should contain 'logits'.")

        self.trainer.train()
        self.trainer.save_model(self.output_dir)


class TestColBERT(TestCase):
    def setUp(self):
        # Create mock objects
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.Linear(768, 256)
        self.colbert = ColBERT(
            model=self.model, tokenizer=self.tokenizer, linear_layer=self.linear_layer, max_length=128, device='cpu'
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
            neg_input_ids=pos_input_ids,
            neg_attention_mask=pos_attention_mask,
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
