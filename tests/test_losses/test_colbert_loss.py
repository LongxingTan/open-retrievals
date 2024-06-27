import unittest

import torch
import torch.nn as nn

from src.retrievals.losses.colbert_loss import ColbertLoss


class TestColbertLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.temperature = 0.05
        self.use_inbatch_negative = True
        self.loss_fn = ColbertLoss(
            criterion=self.criterion,
            temperature=self.temperature,
            use_inbatch_negative=self.use_inbatch_negative,
        )
        self.query_embeddings = torch.randn(2, 10, 128)
        self.positive_embeddings = torch.randn(2, 10, 128)
        self.negative_embeddings = torch.randn(2, 10, 128)

    def test_loss_without_negatives(self):
        loss = self.loss_fn(query_embeddings=self.query_embeddings, positive_embeddings=self.positive_embeddings)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        self.assertGreaterEqual(loss.item(), 0)

    def test_loss_with_negatives(self):
        loss = self.loss_fn(
            query_embeddings=self.query_embeddings,
            positive_embeddings=self.positive_embeddings,
            negative_embeddings=self.negative_embeddings,
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        self.assertGreaterEqual(loss.item(), 0)
