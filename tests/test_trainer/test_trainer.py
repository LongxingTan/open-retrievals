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
    PairwiseModel,
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

        train_model = PairwiseModel(self.model, loss_fn=TripletLoss())
        trainer = RetrievalTrainer(
            model=train_model,
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


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, **kwargs):
        return MockOutput(logits=self.fc(kwargs['input_ids']))


class MockOutput:
    def __init__(self, logits):
        self.logits = logits


class TestDistilTrainer(TestCase):
    def setUp(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = MockModel().to(self.device)
        self.teacher_model = MockModel().to(self.device)
        self.temperature = 1.0
        self.trainer = DistilTrainer(model=self.model, teacher_model=self.teacher_model, temperature=self.temperature)

        self.input_ids = torch.randn(2, 10).to(self.device)
        self.inputs = {'input_ids': self.input_ids}

    def test_compute_loss(self):
        loss, outputs = self.trainer.compute_loss(self.model, self.inputs, return_outputs=True)

        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.dim(), 0)

        self.assertTrue(hasattr(outputs, 'logits'))
        self.assertEqual(outputs.logits.shape, torch.Size([2, 10]))

    def test_get_teacher_probabilities(self):
        teacher_scores = torch.randn(2, 10)  # Batch size of 2, feature size of 10
        student_shape = (2, 10)  # Batch size of 2, feature size of 10

        teacher_probs = self.trainer._get_teacher_probabilities(teacher_scores, student_shape)
        self.assertEqual(teacher_probs.shape, torch.Size(student_shape))
