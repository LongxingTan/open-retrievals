from unittest import TestCase

import torch

from src.retrievals.losses.cosent import CoSentLoss


class CoSentLossTest(TestCase):
    def test_forward(self):
        y_true = torch.randint(0, 2, (8, 1)).float()
        y_pred = torch.randn(8, 10)
        loss_fn = CoSentLoss()
        loss = loss_fn(y_true, y_pred)
        print(loss)
