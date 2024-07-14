"""Matryoshka Representation Learning"""

from typing import List, Optional

import torch
from torch import nn


class MRLLayer(nn.Module):
    def __init__(self, mrl_nested_dim: List[int]):
        super().__init__()
        # self.projection = nn.Linear()
        self.mrl_nested_dim = mrl_nested_dim

    def forward(self, x):
        # x = self.projection(x)
        nested_x = []
        for dim in self.mrl_nested_dim:
            nested_x.append(x[:, :dim])
        return nested_x


class MRLLoss(nn.Module):
    """
    https://arxiv.org/pdf/2205.13147
    """

    def __init__(self, criterion: nn.Module, mrl_nested_dim: List[int]):
        super().__init__()
        self.criterion = criterion
        self.mrl_nested_dim = mrl_nested_dim
        self.query_mrl = MRLLayer(self.mrl_nested_dim)
        self.positive_mrl = MRLLayer(self.mrl_nested_dim)
        self.negative_mrl = MRLLayer(self.mrl_nested_dim)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: Optional[torch.Tensor] = None,
    ):
        query_mrl_embed = self.query_mrl(query_embeddings)
        positive_mrl_embed = self.positive_mrl(pos_embeddings)
        if neg_embeddings is not None:
            negative_mrl_embed = self.negative_mrl(neg_embeddings)
        else:
            negative_mrl_embed = [None] * len(positive_mrl_embed)

        loss = torch.tensor(0, device=query_embeddings.device)
        for query_embed, positive_embed, negative_embed in zip(query_mrl_embed, positive_mrl_embed, negative_mrl_embed):
            loss += self.criterion(query_embed, positive_embed, negative_embed)

        loss = loss / len(self.mrl_nested_dim)

        return loss
