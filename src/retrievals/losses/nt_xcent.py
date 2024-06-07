from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


class NTXcent(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature=0.05,
    ):
        super().__init__()
        self.criterion = criterion
        self.temperature = temperature

    def forward(self, x):
        similarity = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        similarity[torch.eye(x.size(0)).bool()] = float("-inf")
        similarity = similarity / self.temperature

        target = torch.arange(x.size(0))
        target[0::2] += 1
        target[1::2] -= 1

        return F.cross_entropy(similarity, target, reduction="mean")
