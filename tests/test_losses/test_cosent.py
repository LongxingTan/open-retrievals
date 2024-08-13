from unittest import TestCase

import torch

from src.retrievals.losses.cosent import CoSentLoss


class CoSentLossTest(TestCase):
    def test_forward(self):
        embed1 = torch.rand(2, 11)
        embed2 = torch.rand(2, 11)
        labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        loss_fn = CoSentLoss()
        loss = loss_fn(embed1, embed2, labels)
        print(loss)
        self.assertTrue(loss.item() >= 0, "Loss should be non-negative")
