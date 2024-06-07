from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


class MarginMSELoss(nn.Module):
    def __init__(
        self,
        scale: float = 1.0,
    ):
        super(MarginMSELoss, self).__init__()
        self.scale = scale
        self.criterion = nn.MSELoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        labels=None,
    ):
        scores_pos = (query_embeddings * positive_embeddings).sum(dim=-1) * self.scale
        scores_neg = (query_embeddings * negative_embeddings).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.criterion(margin_pred, labels)
