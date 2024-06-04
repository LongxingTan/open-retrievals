"""
`Representation Learning with Contrastive Predictive Coding
<https://arxiv.org/abs/1807.03748v2>`_
"""

import logging
from typing import Callable, Literal, Optional, Union

import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCE(nn.Module):
    """
    https://github.com/RElbers/info-nce-pytorch
    """

    def __init__(
        self,
        criterion: Union[nn.Module, Callable, None] = nn.CrossEntropyLoss(label_smoothing=0.05),
        temperature: float = 0.05,
        use_inbatch_negative: bool = True,
        negative_mode: Literal['paired', 'unpaired'] = "unpaired",
        train_group_size: int = 1,
    ):
        super().__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative
        self.negative_mode = negative_mode
        self.train_group_size = train_group_size

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ):
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)
        device = query_embeddings.device
        if negative_embeddings is None:
            if self.negative_mode == 'unpaired':
                logits = query_embeddings @ positive_embeddings.transpose(-2, -1)
                labels = torch.arange(logits.size(0), dtype=torch.long, device=device)
                loss = self.criterion(logits / self.temperature, labels)
            else:
                logits1 = query_embeddings @ positive_embeddings.transpose(-2, -1)
                logits2 = logits1.T
                labels = torch.arange(logits1.size(0), dtype=torch.long, device=device)
                loss = (
                    self.criterion(logits1 / self.temperature, labels)
                    + self.criterion(logits2 / self.temperature, labels)
                ) / 2
            return loss
        else:
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)
            if self.use_inbatch_negative:
                logits = torch.cat([positive_embeddings, negative_embeddings], dim=0)
                similarity = query_embeddings @ logits.transpose(-2, -1)
                similarity = similarity / self.temperature
                similarity = similarity.view(similarity.size(0), -1)

                labels = torch.arange(similarity.size(0), device=query_embeddings.device, dtype=torch.long)
                labels = labels * self.train_group_size
            else:
                logits = torch.cat([positive_embeddings, negative_embeddings], dim=0)
                logits = logits.view(query_embeddings.size(0), self.train_group_size, -1)
                similarity = query_embeddings[:, None, :] @ logits.transpose(-2, -1)
                similarity = similarity.squeeze(1) / self.temperature
                similarity = similarity.view(query_embeddings.size(0), -1)
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=query_embeddings.device)

            return self.criterion(similarity, labels)
