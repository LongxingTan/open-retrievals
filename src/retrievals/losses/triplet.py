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
        **kwargs
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
        margin: float = 0.0,
    ):
        if margin:
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
        loss = -torch.log(torch.sigmoid(similarity_diff) + self.margin).mean()
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


class TripletRankingLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, use_inbatch_neg: bool = True, symmetric=False):
        super().__init__()
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.symmetric = symmetric
        # self.similarity_fn = F.cosine_similarity
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_embedding, pos_embedding: torch.Tensor, neg_embedding: torch.Tensor):

        if self.use_inbatch_neg:
            document_embedding = torch.concat([pos_embedding, neg_embedding], dim=0)
            scores = self.similarity_fn(query_embedding, document_embedding) / self.temperature
            labels = torch.arange(0, scores.size(0), device=scores.device)

        else:
            negative_embeddings = neg_embedding.view(query_embedding.size(0), -1, query_embedding.size(1))
            similarity = query_embedding.unsqueeze(1) @ pos_embedding.unsqueeze(2)
            negative_similarity = query_embedding.unsqueeze(1) @ negative_embeddings.permute(0, 2, 1)
            similarity = torch.cat([similarity.squeeze(1), negative_similarity.squeeze(1)], dim=1)
            scores = similarity / self.temperature
            labels = torch.zeros(query_embedding.size(0), dtype=torch.long, device=query_embedding.device)

        return self.loss_fn(scores, labels)

    def similarity_fn(self, query_embedding, document_embedding):
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        document_embedding = F.normalize(document_embedding, p=2, dim=-1)
        return torch.mm(query_embedding, document_embedding.transpose(0, 1))
