from unittest import TestCase

import torch

from src.retrievals.losses.dice import DiceLoss


class DiceLossTest(TestCase):
    def test_forward(self):
        pred = torch.randn(5, 3, 256, 256)
        target = torch.randn(5, 3, 256, 256)
        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)
        print(loss)
