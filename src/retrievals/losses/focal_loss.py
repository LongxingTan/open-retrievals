"""
`Focal Loss for Dense Object Detection
<https://arxiv.org/abs/1708.02002>`_
"""

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, eps: float = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        logp = self.ce(inputs, labels)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
