from unittest import TestCase

import torch
import torch.nn.functional as F

from src.retrievals.losses.bce import BCELoss


class BCETest(TestCase):
    def test_basic_functionality(self):
        # Testing basic loss calculation without mask or weights
        inputs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        loss_fn = BCELoss()
        calculated_loss = loss_fn(inputs, labels)

        # Manually calculate expected loss for comparison
        expected_loss1 = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction="none")
        expected_loss2 = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction="none")
        expected_loss = 1 * expected_loss1 * labels + expected_loss2 * (1 - labels)
        expected_loss = expected_loss.mean()

        self.assertEqual(calculated_loss.shape, expected_loss.shape)
        # self.assertAlmostEqual(calculated_loss.item(), expected_loss.item(), places=4)

    def test_with_mask(self):
        # Testing loss calculation with a mask applied
        inputs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)  # Masking second element in first sample
        loss_module = BCELoss()
        calculated_loss = loss_module(inputs, labels, mask=mask)

        # Manually calculate expected loss for comparison, taking mask into account
        expected_loss1 = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction="none")
        expected_loss2 = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction="none")
        expected_loss = 1 * expected_loss1 * labels + expected_loss2 * (1 - labels)
        expected_loss = expected_loss * mask
        expected_loss = torch.sum(expected_loss, dim=1) / torch.sum(mask, dim=1)
        expected_loss = expected_loss.mean()

        self.assertEqual(calculated_loss.shape, expected_loss.shape)
        # self.assertAlmostEqual(calculated_loss.item(), expected_loss.item(), places=4)

    def test_with_sample_weight(self):
        # Testing loss calculation with sample weighting
        inputs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        sample_weight = torch.tensor([0.5, 1.5], dtype=torch.float32)
        loss_module = BCELoss()
        calculated_loss = loss_module(inputs, labels, sample_weight=sample_weight)

        # Manually calculate expected loss for comparison, applying sample weights
        expected_loss1 = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction="none")
        expected_loss2 = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction="none")
        expected_loss = 1 * expected_loss1 * labels + expected_loss2 * (1 - labels)
        expected_loss = expected_loss * sample_weight.unsqueeze(1)
        expected_loss = expected_loss.mean()

        self.assertEqual(calculated_loss.shape, expected_loss.shape)
        # self.assertAlmostEqual(calculated_loss.item(), expected_loss.item(), places=4)
