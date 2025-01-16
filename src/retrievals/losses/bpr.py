from typing import Optional

import torch

from .base import Base


class BPRLoss(Base):
    """Bayesian Personalized Ranking"""

    def __init__(self, negatives_cross_device: bool = False):
        super(BPRLoss, self).__init__(negatives_cross_device)

    def forward(
        self, query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        return -(query_embeddings - positive_embeddings).sigmoid().log().sum()
