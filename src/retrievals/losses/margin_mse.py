from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from .base import Base


class MarginMSELoss(Base):
    def __init__(self, scale: float = 1.0, negatives_cross_device: bool = False, **kwargs):
        super(MarginMSELoss, self).__init__(negatives_cross_device)
        self.scale = scale
        self.criterion = nn.MSELoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            negative_embeddings = self._dist_gather_tensor(negative_embeddings)
            labels = self._dist_gather_tensor(labels)

        scores_pos = (query_embeddings * positive_embeddings).sum(dim=-1) * self.scale
        scores_neg = (query_embeddings * negative_embeddings).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.criterion(margin_pred, labels)
