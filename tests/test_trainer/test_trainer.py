import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from src.retrievals import (
    AutoModelForEmbedding,
    AutoModelForRanking,
    RerankCollator,
    RetrievalCollator,
)
from src.retrievals.losses import TripletLoss
from src.retrievals.trainer.trainer import (
    DistilTrainer,
    RerankTrainer,
    RetrievalTrainer,
)


class PseudoDataset(Dataset):
    def __init__(self):
        self.examples = [
            {'query': 'how are you', 'positive': 'fine', 'negative': 'and you?'},
            {'query': 'hallo?', 'positive': 'what is your problem', 'negative': 'I am a doctor'},
            {'query': 'how are you doing', 'positive': 'survive', 'negative': 'he looks like a dog'},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = tempfile.mkdtemp()
    do_train: bool = field(default=True)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(default="cls", metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})


class TrainerTest(TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")
        self.train_dataset = PseudoDataset()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=self.output_dir)
        self.mock_loss_fn = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_trainer(self):
        parser = HfArgumentParser((TrainingArguments))
        training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        trainer = RetrievalTrainer(
            model=self.model.set_train_type('pairwise', loss_fn=TripletLoss()),
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=RetrievalCollator(
                tokenizer=self.tokenizer, keys=['query', 'positive', 'negative'], max_lengths=[32, 64, 64]
            ),
        )
        trainer.train()

    def test_example(self):
        sentences = ["Hello world", "How are you?"]

        sentence_embeddings = self.model.encode(sentences)

        self.assertEqual(sentence_embeddings.shape, torch.Size([2, 384]))


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


class TestRerankTrainer(TestCase):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = "distilbert/distilbert-base-uncased"
        self.model = AutoModelForRanking.from_pretrained(model_name_or_path)
        self.train_dataset = PseudoRerankTrainDataset()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=self.output_dir)
        self.trainer = RerankTrainer(
            model=self.model, train_dataset=self.train_dataset, tokenizer=self.tokenizer, loss_fn=nn.BCEWithLogitsLoss()
        )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_compute_loss(self):
        inputs = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}

        loss, outputs = self.trainer.compute_loss(self.model, inputs, return_outputs=True)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertIn('loss', outputs)

    def test_save_model(self):
        self.trainer._save(output_dir=self.output_dir)


class TestDistilTrainer(TestCase):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = "distilbert-base-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.train_dataset = PseudoRerankTrainDataset()
        self.trainer = DistilTrainer(
            model=self.model,
            teacher_model=self.teacher_model,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
        )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_compute_loss(self):
        inputs = {'input_ids': torch.tensor([[101, 201, 301]]), 'attention_mask': torch.tensor([[1, 1, 1]])}

        loss = self.trainer.compute_loss(self.model, inputs)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_save_model(self):
        self.trainer._save(output_dir=self.output_dir)
