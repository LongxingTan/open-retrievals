from typing import Callable, Optional, Union

import torch
import torch.nn as nn


class PairKL(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable, None] = nn.KLDivLoss(reduction="batchmean"),
        temperature: float = 0.05,
    ):
        super().__init__()
        self.criterion = criterion
        self.temperature = temperature

    def forward(
        self, query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, scores: torch.Tensor, **kwargs
    ):
        similarity = torch.einsum('bn,bn->b', query_embeddings, positive_embeddings)
        similarity = similarity / self.temperature
        similarity = torch.log_softmax(similarity, dim=-1)
        target = torch.softmax(scores / self.temperature, dim=-1)

        return self.criterion(similarity, target)
