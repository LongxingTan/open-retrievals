"""
`FaceNet: A Unified Embedding for Face Recognition and Clustering
<https://arxiv.org/abs/1503.03832>`_
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base


class TripletLoss(Base):
    """
    https://omoindrot.github.io/triplet-loss
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.0,
        use_inbatch_negative: bool = False,
        negatives_cross_device: bool = False,
        **kwargs
    ):
        super().__init__(negatives_cross_device)
        self.temperature = temperature
        self.margin = margin
        self.negatives_cross_device = negatives_cross_device
        self.use_inbatch_negative = use_inbatch_negative

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        margin: float = 0.0,
    ):
        if margin:
            self.set_margin(margin=margin)

        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            negative_embeddings = self._dist_gather_tensor(negative_embeddings)

        pos_similarity = torch.cosine_similarity(query_embeddings, positive_embeddings, dim=-1)
        pos_similarity = pos_similarity / self.temperature
        neg_similarity = torch.cosine_similarity(
            query_embeddings.unsqueeze(1),
            negative_embeddings.unsqueeze(0),
            dim=-1,
        )
        neg_similarity = neg_similarity / self.temperature
        similarity_diff = pos_similarity.unsqueeze(1) - neg_similarity
        if mask is not None:
            similarity_diff = similarity_diff * mask
            mask_sum = mask.sum() if mask.sum() > 0 else 1  # Avoid division by zero
            loss = -torch.log(torch.sigmoid(similarity_diff) + self.margin).sum() / mask_sum
        else:
            loss = -torch.log(torch.sigmoid(similarity_diff) + self.margin).mean()

        return loss

    def set_margin(self, margin: float):
        self.margin = margin


class TripletCosineSimilarity(nn.Module):
    def __init__(self, temperature: float = 1.0, margin: float = 0.50):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = lambda x, y: F.pairwise_distance(x, y, p=2)

    def forward(
        self,
        query_embedding: torch.Tensor,
        pos_embedding: torch.Tensor,
        neg_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        distance_pos = self.distance_metric(query_embedding, pos_embedding)
        distance_neg = self.distance_metric(query_embedding, neg_embedding)

        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()


class TripletRankingLoss(nn.Module):
    """Similar to sentence-transformers MultiNegativesRankingLoss"""

    def __init__(self, temperature: float = 0.05, use_inbatch_negative: bool = True, symmetric=False):
        super().__init__()
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative
        self.symmetric = symmetric
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self, query_embedding, positive_embedding: torch.Tensor, negative_embedding: Optional[torch.Tensor] = None
    ):

        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        positive_embedding = F.normalize(positive_embedding, p=2, dim=-1)
        if negative_embedding is not None:
            negative_embedding = F.normalize(negative_embedding, p=2, dim=-1)

        if self.use_inbatch_negative:
            if negative_embedding is not None:
                positive_embedding = torch.concat([positive_embedding, negative_embedding], dim=0)
            scores = torch.mm(query_embedding, positive_embedding.transpose(0, 1)) / self.temperature
            labels = torch.arange(0, scores.size(0), device=scores.device)

        else:
            similarity = query_embedding.unsqueeze(1) @ positive_embedding.unsqueeze(2)
            if negative_embedding is not None:
                negative_embedding = negative_embedding.view(query_embedding.size(0), -1, query_embedding.size(1))
                negative_similarity = query_embedding.unsqueeze(1) @ negative_embedding.permute(0, 2, 1)
                similarity = torch.cat([similarity.squeeze(1), negative_similarity.squeeze(1)], dim=1)
            scores = similarity / self.temperature
            labels = torch.zeros(query_embedding.size(0), dtype=torch.long, device=query_embedding.device)

        return self.criterion(scores, labels)
