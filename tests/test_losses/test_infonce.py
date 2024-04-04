from unittest import TestCase

import torch
import torch.nn as nn

from src.retrievals.losses.infonce import InfoNCE

from .test_losses_common import LossTesterMixin


class InfoNCETest(TestCase, LossTesterMixin):
    def setUp(self) -> None:
        self.query_embedding = torch.rand(2, 11)
        self.positive_embedding = torch.rand(2, 11)
        self.negative_embedding = torch.rand(2, 11)

    def test_infonce_pair(self):
        loss_fn = InfoNCE(criterion=nn.CrossEntropyLoss())
        loss = loss_fn(self.query_embedding, self.positive_embedding)

        self.assertEqual(loss.shape, torch.Size([]))

    def test_infonce_triplet(self):
        loss_fn = InfoNCE(criterion=nn.CrossEntropyLoss(), negative_mode='paired')
        paired_triplet_loss = loss_fn(
            self.query_embedding,
            self.positive_embedding,
            self.negative_embedding,
        )
        self.assertEqual(paired_triplet_loss.shape, torch.Size([]))
        loss_fn = InfoNCE(criterion=nn.CrossEntropyLoss(), negative_mode='unpaired')
        unpaired_triplet_loss = loss_fn(self.query_embedding, self.positive_embedding, self.negative_embedding)
        self.assertEqual(unpaired_triplet_loss.shape, torch.Size([]))
