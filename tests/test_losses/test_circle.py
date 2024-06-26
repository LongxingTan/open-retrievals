from unittest import TestCase

import torch

from src.retrievals.losses.circle import MultiLabelCircleLoss


class MultiLabelCircleLossTest(TestCase):
    def test_forward(self):
        loss_fn = MultiLabelCircleLoss()
        logits = torch.randn(4, 13)
        labels = torch.randint(0, 2, (4, 13)).float()
        loss = loss_fn(logits, labels)
        print(loss)
