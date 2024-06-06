from typing import Callable, Optional, Union

import torch
from torch import nn


class MoCoLoss(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature=0.05,
    ):
        super(MoCoLoss, self).__init__()
        self.criterion = criterion
        self.temperature = temperature

    def forward(self, q, k):
        batch_size = q.shape[0]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # Positive logits: N x 1
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: N x K
        neg_logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([pos_logits, neg_logits], dim=1)

        logits = logits / self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)

        return loss
