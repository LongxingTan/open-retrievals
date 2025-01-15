from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from .base import Base


class PairKL(Base):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable, None] = nn.KLDivLoss(reduction="batchmean"),
        temperature: float = 0.05,
        negatives_cross_device: bool = False,
        **kwargs
    ):
        super().__init__(negatives_cross_device)
        self.criterion = criterion
        self.temperature = temperature

    def forward(
        self, query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, scores: torch.Tensor, **kwargs
    ):
        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            scores = self._dist_gather_tensor(scores)

        similarity = torch.einsum('bn,bn->b', query_embeddings, positive_embeddings)
        similarity = similarity / self.temperature
        similarity = torch.log_softmax(similarity, dim=-1)
        target = torch.softmax(scores / self.temperature, dim=-1)

        return self.criterion(similarity, target)
