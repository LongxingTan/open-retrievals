import unittest

import torch
import torch.nn.functional as F

from src.retrievals.losses.simcse import SimCSE

from .test_losses_common import LossTesterMixin


class SimCSETest(unittest.TestCase, LossTesterMixin):
    def setUp(self) -> None:
        self.loss_fn = SimCSE(criterion=F.cross_entropy)
        self.y_pred = torch.rand(2, 11)
        self.y_ture = torch.rand(2, 11)

    def test_loss(self):
        loss = self.loss_fn(self.y_pred, self.y_ture)
        self.assertEqual(loss.shape, torch.Size([]))
