import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking"""

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ):
        return -(query_embeddings - positive_embeddings).sigmoid().log().sum()
