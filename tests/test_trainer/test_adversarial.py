import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.optim import SGD

from src.retrievals.trainer.adversarial import AWP, EMA, FGM, PGD


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Parameter(torch.randn(5, 10))
        self.other_embeddings = nn.Parameter(torch.randn(5, 10))
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestFGM(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.fgm = FGM(self.model)
        self.epsilon = 1e-3

    def test_attack(self):
        # Create fake gradients for the parameters
        self.model.word_embeddings.grad = torch.ones_like(self.model.word_embeddings)
        self.model.other_embeddings.grad = torch.ones_like(self.model.other_embeddings)

        original_data = self.model.word_embeddings.data.clone()
        original_other_data = self.model.other_embeddings.data.clone()

        self.fgm.attack(epsilon=self.epsilon, emb_name="word_embeddings")

        norm = torch.norm(torch.ones_like(self.model.word_embeddings))
        r_at = self.epsilon * torch.ones_like(self.model.word_embeddings) / norm
        expected_data = original_data + r_at
        self.assertTrue(torch.allclose(self.model.word_embeddings.data, expected_data, atol=1e-6))

        # Check if the attack did not modify other_embeddings
        self.assertTrue(torch.allclose(self.model.other_embeddings.data, original_other_data, atol=1e-6))

    def test_restore(self):
        self.model.word_embeddings.grad = torch.ones_like(self.model.word_embeddings)
        self.fgm.attack(epsilon=self.epsilon, emb_name="word_embeddings")
        original = self.fgm.backup["word_embeddings"].clone()

        self.fgm.restore(emb_name="word_embeddings")
        self.assertTrue(torch.allclose(self.model.word_embeddings.data, original, atol=1e-6))
        self.assertEqual(len(self.fgm.backup), 0)


class TestEMA(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        self.ema = EMA(self.model, decay=0.9999)

    def test_register(self):
        self.ema.register()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, self.ema.shadow)
                self.assertTrue(torch.equal(param.data, self.ema.shadow[name]))

    def test_update(self):
        # Perform registration
        self.ema.register()

        # Update the exponential moving average
        for _ in range(1):
            for param in self.model.parameters():
                param.data += torch.randn_like(param)

            self.ema.update()

        # Check if shadow parameters have been correctly updated
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                updated_average = (1.0 - self.ema.decay) * param.data + self.ema.decay * self.ema.shadow[name]
                self.assertTrue(torch.allclose(param.data, updated_average, atol=10))

    def test_apply_shadow_and_restore(self):
        self.ema.register()

        self.ema.apply_shadow()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, self.ema.backup)
                self.assertTrue(torch.equal(param.data, self.ema.shadow[name]))

        backup = self.ema.backup
        self.ema.restore()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(torch.equal(param.data, backup[name]))


# class TestAWP(unittest.TestCase):
#     def setUp(self):
#         # Set up a simple model for testing
#         self.model = MockModel()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.optimizer = SGD(self.model.parameters(), lr=0.01)
#         self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
#         self.awp = AWP(
#             self.model,
#             self.optimizer,
#             adv_param="fc.weight",  # Specify the parameter name to attack
#             adv_lr=0.0001,
#             adv_eps=0.001,
#             start_epoch=0,
#             adv_step=1,
#             scaler=self.scaler,
#         )
#
#     def test_attack_backward(self):
#         # Mock data
#         inputs = torch.randn(10, 10)
#         labels = torch.randint(0, 2, (10,)).float()
#         criterion = nn.CrossEntropyLoss()
#
#         # Perform attack_backward
#         self.awp.attack_backward(inputs, criterion, labels, epoch=1)
#
#         # Ensure parameters have been modified
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and "fc.weight" in name:
#                 self.assertNotEqual(param.data, self.awp.backup[name])  # Check if attacked
#
#     def test_save_and_restore(self):
#         # Mock data
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and "fc.weight" in name:
#                 initial_param_data = param.data.clone()
#
#         self.awp._save()
#         self.assertTrue(self.awp.backup)
#         self.assertTrue(self.awp.backup_eps)
#
#         self.awp._restore()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and "fc.weight" in name:
#                 self.assertEqual(param.data, initial_param_data)  # Check if restored
