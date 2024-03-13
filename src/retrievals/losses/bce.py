import torch
import torch.nn.functional as F
from torch import nn


class BCELoss(nn.Module):
    def __init__(self, temperature: float = 0.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, inputs, labels, mask=None, sample_weight=None, class_weight=None):
        bce1 = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction="none")
        bce2 = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction="none")
        bce = 1 * bce1 * labels + bce2 * (1 - labels)
        if mask is not None:
            # mask = torch.where(targets >= 0, torch.ones_like(bce), torch.zeros_like(bce))
            bce = bce * mask

        if sample_weight is not None:
            bce = bce * sample_weight.unsqueeze(1)
        loss = torch.sum(bce, dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss
