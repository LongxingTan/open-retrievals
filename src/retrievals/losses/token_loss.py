import logging
from typing import Callable, Literal, Optional, Union

import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


class TokenLoss(nn.Module):
    def __init__(
        self,
        check_loc,
        criterion: Union[nn.Module, Callable, None] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
    ):
        super().__init__()
        self.check_loc = check_loc
        self.criterion = criterion

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        _, max_indices = torch.max(labels, dim=1)
        # shift the targets such that output n predicts token n+1
        predict_indices = max_indices - 1
        logits = [logits[i, predict_indices[i], :] for i in range(logits.shape[0])]
        logits = torch.stack(logits, dim=0)
        scores = logits[:, self.check_loc]

        grouped_scores = scores.contiguous().view(batch_size, -1)
        target = torch.zeros(batch_size, device=grouped_scores.device, dtype=torch.long)
        loss = self.criterion(grouped_scores, target)
        return loss
