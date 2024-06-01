from typing import Optional

import torch
import torch.nn as nn


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.inf = 1e12

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logits = (1 - 2 * labels) * logits
        logits_neg = logits - labels * self.inf
        logits_pos = logits - (1 - labels) * self.inf
        zeros = torch.zeros_like(logits[..., :1])
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = neg_loss + pos_loss
        if mask is not None:
            loss = loss / (mask.sum(-1).float() + 1e-12)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss
