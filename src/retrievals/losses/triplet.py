"""
`FaceNet: A Unified Embedding for Face Recognition and Clustering
<https://arxiv.org/abs/1503.03832>`_
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    https://omoindrot.github.io/triplet-loss
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.0,
        negatives_cross_device: bool = False,
        batch_hard: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.negatives_cross_device = negatives_cross_device
        self.batch_hard = batch_hard
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Cannot do negatives_cross_device without distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor,
        margin: Optional[float] = None,
    ):
        if margin is not None:
            # dynamic margin
            self.set_margin(margin=margin)

        if self.negatives_cross_device:
            pos_embeddings = self._dist_gather_tensor(pos_embeddings)
            neg_embeddings = self._dist_gather_tensor(neg_embeddings)

        pos_similarity = torch.cosine_similarity(query_embeddings, pos_embeddings, dim=-1)
        pos_similarity = pos_similarity / self.temperature
        neg_similarity = torch.cosine_similarity(
            query_embeddings.unsqueeze(1),
            neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        neg_similarity = neg_similarity / self.temperature
        similarity_diff = pos_similarity.unsqueeze(1) - neg_similarity
        loss = -torch.log(torch.sigmoid(similarity_diff)).mean()
        return loss

    def set_margin(self, margin: float):
        self.margin = margin

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class TripletCosineSimilarity(nn.Module):
    def __init__(self, temperature: float = 1.0, margin: float = 0.50):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = lambda x, y: F.pairwise_distance(x, y, p=2)

    def forward(self, query_embedding: torch.Tensor, pos_embedding: torch.Tensor, neg_embedding: torch.Tensor):
        distance_pos = self.distance_metric(query_embedding, pos_embedding)
        distance_neg = self.distance_metric(query_embedding, neg_embedding)

        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()
