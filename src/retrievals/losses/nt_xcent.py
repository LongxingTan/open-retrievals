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
        xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")

        target = torch.arange(x.size(0))
        target[0::2] += 1
        target[1::2] -= 1

        # Standard cross entropy loss
        return F.cross_entropy(xcs / self.temperature, target, reduction="mean")
