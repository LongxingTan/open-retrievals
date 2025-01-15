import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from src.retrievals.trainer.custom_trainer import (
    MetricsTracker,
    NLPTrainer,
    TrainingArguments,
)


class MockDataset(Dataset):
    def __init__(self, size=100, seq_length=32):
        self.size = size
        self.seq_length = seq_length
        self.input_ids = torch.randint(0, 1000, (size, seq_length))
        self.attention_mask = torch.ones((size, seq_length))
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


class MockPreTrainedModel(PreTrainedModel):
    def __init__(self):
        config = PretrainedConfig()
        super().__init__(config=config)  # If you don't need a config, use `None`
        self.encoder = nn.Linear(32, 768)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.encoder(input_ids.float())  # Ensure input is float
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return type('ModelOutput', (), {'loss': loss, 'logits': logits, 'hidden_states': hidden_states})


class TestNLPTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize components
        self.model = MockPreTrainedModel()
        self.train_dataset = MockDataset(size=100)
        self.eval_dataset = MockDataset(size=50)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=16, shuffle=False)

        self.args = TrainingArguments(
            epochs=1,
            learning_rate=1e-4,
            fp16=False,
            logging_steps=10,
            evaluation_steps=50,
            use_fgm=False,
            use_awp=False,
            use_ema=False,
        )

        self.optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate)

        self.trainer = NLPTrainer(
            model=self.model,
            args=self.args,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            optimizer=self.optimizer,
        )

    def test_initialization(self):
        """Test if trainer initializes correctly"""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.train_dataloader)
        self.assertEqual(self.trainer.device, "cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step(self):
        """Test single training step"""
        batch = next(iter(self.train_dataloader))
        metrics = self.trainer.train_step(batch)

        self.assertIn('loss', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertGreater(metrics['loss'], 0)

    def test_evaluate(self):
        """Test evaluation loop"""
        eval_metrics = self.trainer.evaluate()

        self.assertIn('eval_loss', eval_metrics)
        self.assertIsInstance(eval_metrics['eval_loss'], float)

    def test_full_training_loop(self):
        """Test complete training loop"""
        try:
            self.trainer.train()
        except Exception as e:
            self.fail(f"Training loop failed with error: {str(e)}")

    def test_save_load_model(self):
        """Test model saving and loading"""
        temp_dir = tempfile.mkdtemp()

        try:
            save_path = f"{temp_dir}/test_model.pt"
            self.trainer.save_model(save_path)

            original_state_dict = self.trainer.model.state_dict()
            self.trainer.load_model(save_path)
            loaded_state_dict = self.trainer.model.state_dict()

            for key in original_state_dict:
                self.assertTrue(torch.equal(original_state_dict[key], loaded_state_dict[key]))

        finally:
            shutil.rmtree(temp_dir)

    def test_fp16_training(self):
        """Test FP16 training if cuda is available"""
        if torch.cuda.is_available():
            self.args.fp16 = True
            self.trainer = NLPTrainer(
                model=self.model,
                args=self.args,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                optimizer=self.optimizer,
            )

            # Cast model and data to fp16 if needed
            self.model = self.model.half() if self.args.fp16 else self.model
            batch = next(iter(self.train_dataloader))
            metrics = self.trainer.train_step(batch)

            self.assertIn('loss', metrics)
            self.assertIsInstance(metrics['loss'], float)

    def test_gradient_accumulation(self):
        """Test gradient accumulation steps"""
        self.args.gradient_accumulation_steps = 2
        trainer = NLPTrainer(
            model=self.model,
            args=self.args,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            optimizer=self.optimizer,
        )

        # Train for a few steps
        for i, batch in enumerate(self.train_dataloader):
            trainer.train_step(batch)
            if i >= 4:  # Test for 4 steps
                break

    def test_metrics_tracker(self):
        """Test metrics tracking functionality"""
        metrics_tracker = MetricsTracker()

        # Update with some dummy metrics
        metrics_tracker.update({'loss': 1.0, 'accuracy': 0.8}, batch_size=16)
        metrics_tracker.update({'loss': 0.8, 'accuracy': 0.9}, batch_size=16)

        final_metrics = metrics_tracker.get_metrics()

        self.assertIn('loss', final_metrics)
        self.assertIn('accuracy', final_metrics)
        self.assertTrue(0 <= final_metrics['loss'] <= 1.0)
        self.assertTrue(0 <= final_metrics['accuracy'] <= 1.0)
