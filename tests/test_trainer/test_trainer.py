import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional
from unittest import TestCase

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, HfArgumentParser

from src.retrievals import AutoModelForEmbedding, TripletCollator
from src.retrievals.losses import TripletLoss
from src.retrievals.trainer.custom_trainer import CustomTrainer
from src.retrievals.trainer.trainer import RetrievalTrainer


class PseudoDataset(Dataset):
    def __init__(self):
        self.examples = [
            {'query': 'how are you', 'pos': 'fine', 'neg': 'and you?'},
            {'query': 'hallo?', 'pos': 'what is your problem', 'neg': 'I am a doctor'},
            {'query': 'how are you doing', 'pos': 'survive', 'neg': 'he looks like a dog'},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class TrainerTest(TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = AutoModelForEmbedding(model_name_or_path, pooling_method="cls")
        self.train_dataset = PseudoDataset()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_trainer(self):
        # training_args = TrainingArguments(
        #     output_dir=self.output_dir,
        #     do_train=True,
        #     learning_rate=1e-5,
        #     per_device_train_batch_size=1,
        #     num_train_epochs=1,
        # )

        @dataclass
        class TrainingArguments(transformers.TrainingArguments):
            output_dir: str = self.output_dir
            do_train: bool = True
            num_train_epochs: int = 1
            per_device_train_batch_size: int = 1
            remove_unused_columns: bool = False
            negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
            temperature: Optional[float] = field(default=0.02)
            fix_position_embedding: bool = field(
                default=False, metadata={"help": "Freeze the parameters of position embeddings"}
            )
            sentence_pooling_method: str = field(
                default="cls", metadata={"help": "the pooling method, should be cls or mean"}
            )
            normalized: bool = field(default=True)
            use_inbatch_neg: bool = field(
                default=True, metadata={"help": "Freeze the parameters of position embeddings"}
            )

        parser = HfArgumentParser((TrainingArguments))
        training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        trainer = RetrievalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=TripletCollator(tokenizer=self.tokenizer, max_length=22),
            loss_fn=TripletLoss(),
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
