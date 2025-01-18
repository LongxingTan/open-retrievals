import unittest

import torch

from src.retrievals.losses.pearson_loss import PearsonLoss


class TestPearsonLoss(unittest.TestCase):
    def test_pearson_loss(self):
        pearson_loss = PearsonLoss()

        # Test case 1: Perfect positive correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_loss = 0.0  # Since the correlation is 1, the loss should be 1 - 1 = 0
        self.assertAlmostEqual(pearson_loss(x, y).item(), expected_loss, places=6)

        # Test case 2: Perfect negative correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        expected_loss = 2.0  # Since the correlation is -1, the loss should be 1 - (-1) = 2
        self.assertAlmostEqual(pearson_loss(x, y).item(), expected_loss, places=6)

        # Test case 3: No correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 3.0, 2.0, 3.0, 2.0])
        # The expected loss should be 1 - r, where r is the Pearson correlation coefficient
        # For this specific case, the correlation is 0, so the loss should be 1 - 0 = 1
        expected_loss = 1.0
        self.assertAlmostEqual(pearson_loss(x, y).item(), expected_loss, places=6)

        # Test case 4: Random tensors
        x = torch.rand(10)
        y = torch.rand(10)
        loss = pearson_loss(x, y)
        self.assertEqual(loss.shape, torch.Size([]))  # Check that the output is a scalar
