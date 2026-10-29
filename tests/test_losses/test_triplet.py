from unittest import TestCase

import torch

from src.retrievals.losses.triplet import TripletCosineSimilarity, TripletLoss


class TripletLossTest(TestCase):
    def test_forward(self):
        batch_size = 8
        embedding_dim = 128
        query_embeddings = torch.randn(batch_size, embedding_dim)
        pos_embeddings = torch.randn(batch_size, embedding_dim)
        neg_embeddings = torch.randn(batch_size, embedding_dim)

        loss_fn = TripletLoss()
        loss = loss_fn(query_embeddings, pos_embeddings, neg_embeddings)
        print("Triplet Loss:", loss.item())

    def test_margin_and_mask(self):
        query = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.0, 1.0]], requires_grad=True)
        negative = torch.tensor([[1.0, 0.0]])
        criterion = TripletLoss(temperature=1.0, margin=0.5)
        loss = criterion(query, positive, negative)
        torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(1.5)))
        loss.backward()
        self.assertLess(positive.grad[0, 0].item(), -0.8)
        masked = criterion(query, positive, negative, mask=torch.zeros(1, 1))
        self.assertEqual(masked.item(), 0.0)

    def test_missing_negatives(self):
        embeddings = torch.ones(1, 2)
        for negatives in (None, torch.empty(0, 2)):
            with self.assertRaisesRegex(ValueError, "explicit negative"):
                TripletLoss()(embeddings, embeddings, negatives)
