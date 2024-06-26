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
