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
from transformers import AutoTokenizer, HfArgumentParser

from src.retrievals import (
    AutoModelForEmbedding,
    AutoModelForRanking,
    RerankCollator,
    TripletCollator,
)
from src.retrievals.losses import TripletLoss
from src.retrievals.trainer.trainer import RerankTrainer, RetrievalTrainer


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

    # @patch("src.retrievals.losses.TripletLoss")
    # def test_compute_loss(self, mock_loss_fn):
    #     inputs = {
    #         "query": torch.tensor([[1.0, 2.0]]),
    #         "pos": torch.tensor([[1.0, 2.0]]),
    #         "neg": torch.tensor([[3.0, 4.0]]),
    #     }
    #     model = MagicMock()
    #     trainer = RetrievalTrainer(loss_fn=mock_loss_fn)
    #     loss = trainer.compute_loss(model, inputs, return_outputs=False)
    #     self.assertIsNotNone(loss)  # or other assertions based on expected behavior
    #     mock_loss_fn.assert_called()

    def test_trainer(self):
        parser = HfArgumentParser((TrainingArguments))
        training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        trainer = RetrievalTrainer(
            model=self.model.set_train_type('pairwise', loss_fn=TripletLoss()),
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=TripletCollator(tokenizer=self.tokenizer, query_max_length=32, document_max_length=128),
        )
        trainer.train()

    # def test_custom_trainer(self):
    #     train_loader = DataLoader(self.train_dataset)
    #     optimizer = AdamW(
    #         self.model.parameters(),
    #         lr=1e-5,
    #     )
    #     trainer = CustomTrainer(model=self.model)
    #
    #     trainer.train(train_loader, optimizer=optimizer, epochs=1, criterion=None)

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

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    # def test_init_with_default_loss_fn(self):
    #     model = None
    #
    #     trainer = RerankTrainer(model)
    #     self.assertIsInstance(trainer.loss_fn, nn.BCEWithLogitsLoss)

    def test_train(self):
        parser = HfArgumentParser((TrainingArguments))
        training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        trainer = RerankTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=RerankCollator(tokenizer=self.tokenizer, max_length=64),
        )
        # trainer.train()
