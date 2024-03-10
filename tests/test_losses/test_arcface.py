from unittest import TestCase

import torch

from src.retrievals.losses.arcface import ArcFaceAdaptiveMarginLoss

from .test_losses_common import LossTesterMixin


class ArcfaceTest(TestCase):
    def setUp(self):
        self.loss_tester = None
        self.loss_fn = ArcFaceAdaptiveMarginLoss(
            in_features=2,
            out_features=1,
        )

    # def test_arcface(self):
    #     loss = self.loss_fn
    #     self.assertEqual(loss.shape, torch.Size([2, 6]))
