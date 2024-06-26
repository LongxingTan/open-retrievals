from unittest import TestCase

import torch

from src.retrievals.losses.r_drop import RDropLoss


class RDropLossTest(TestCase):
    def test_forward(self):
        batch_size = 10  # Should be even to create pairs
        num_classes = 5
        inputs = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        loss_fn = RDropLoss()
        loss = loss_fn(inputs, labels)
        print("RDrop Loss:", loss.item())
