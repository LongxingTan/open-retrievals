"""
`Representation Learning with Contrastive Predictive Coding
<https://arxiv.org/abs/1807.03748v2>`_
"""

import logging
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

logger = logging.getLogger(__name__)


class InfoNCE(Base):
    """
    https://github.com/RElbers/info-nce-pytorch
    """

    def __init__(
        self,
        criterion: Union[nn.Module, Callable, None] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature: float = 0.05,
        use_inbatch_negative: bool = True,
        negatives_cross_device: bool = False,
        negative_mode: Literal['paired', 'unpaired'] = "unpaired",
        **kwargs
    ):
        """
        if not normalized: temperature = 1.0, reset temperature = 1.0 due to using inner product to compute similarity
        if normalized: temperature should be smaller than 1.0 when use cosine similarity. Recommend to set it 0.01-0.1
        """
        super().__init__(negatives_cross_device)
        self.criterion = criterion
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative
        self.negatives_cross_device = negatives_cross_device
        self.negative_mode = negative_mode
        if self.temperature > 0.5:
            logger.error('InfoNCE loss use normalized and inner product by default, temperature should be 0.01 ~ 0.1')

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            negative_embeddings = self._dist_gather_tensor(negative_embeddings)

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)
        device = query_embeddings.device

        if negative_embeddings is None:
            if self.negative_mode == 'unpaired':
                logits = query_embeddings @ positive_embeddings.transpose(-2, -1)
                target = torch.arange(logits.size(0), dtype=torch.long, device=device)
                loss = self.criterion(logits / self.temperature, target)
            else:
                logits1 = query_embeddings @ positive_embeddings.transpose(-2, -1)
                logits2 = logits1.T
                target = torch.arange(logits1.size(0), dtype=torch.long, device=device)
                loss = (
                    self.criterion(logits1 / self.temperature, target)
                    + self.criterion(logits2 / self.temperature, target)
                ) / 2
            return loss
        else:
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

            if self.use_inbatch_negative:
                logits = torch.cat([positive_embeddings, negative_embeddings], dim=0)
                similarity = query_embeddings @ logits.transpose(-2, -1)
                similarity = similarity / self.temperature
                similarity = similarity.view(query_embeddings.size(0), -1)
                target = torch.arange(query_embeddings.size(0), dtype=torch.long, device=device)
            else:
                # negative_embeddings: [batch_size, num_negative_samples, embedding_size]
                negative_embeddings = negative_embeddings.view(query_embeddings.size(0), -1, query_embeddings.size(1))
                similarity = query_embeddings.unsqueeze(1) @ positive_embeddings.unsqueeze(2)
                negative_similarity = query_embeddings.unsqueeze(1) @ negative_embeddings.permute(0, 2, 1)
                # similarity: [batch_size, train_group_sizes]
                similarity = torch.cat([similarity.squeeze(1), negative_similarity.squeeze(1)], dim=1)
                similarity = similarity / self.temperature
                target = torch.zeros(query_embeddings.size(0), dtype=torch.long, device=device)

            if mask is not None:
                similarity = similarity * mask
                mask_sum = mask.sum() if mask.sum() > 0 else 1  # Avoid division by zero
                loss = self.criterion(similarity, target)
                loss = loss.sum() / mask_sum
            else:
                loss = self.criterion(similarity, target)

            return loss
