"""
`SimCSE: Simple Contrastive Learning of Sentence Embeddings
<https://arxiv.org/abs/2104.08821>`_
"""

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .base import Base

logger = logging.getLogger(__name__)


class SimCSE(Base):
    """Cosine similarity = normalize + inner product"""

    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature: float = 0.05,
        dynamic_temperature: bool = False,
        negatives_cross_device: bool = False,
        **kwargs
    ):
        super().__init__(negatives_cross_device)
        self.criterion = criterion
        self.temperature = temperature
        if dynamic_temperature:
            # TODO
            self.temperature = nn.Parameter(torch.tensor(temperature))

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

        similarity = F.cosine_similarity(query_embeddings.unsqueeze(1), positive_embeddings.unsqueeze(0), dim=-1)

        if negative_embeddings is not None:
            neg_similarity = F.cosine_similarity(
                query_embeddings.unsqueeze(1), negative_embeddings.unsqueeze(0), dim=-1
            )
            similarity = torch.cat([similarity, neg_similarity], dim=1)

        similarity = similarity / self.temperature
        if mask is not None:
            # Mask the similarity matrix, zero out the masked positions
            similarity = similarity * mask
            mask_sum = mask.sum() if mask.sum() > 0 else 1  # Avoid division by zero
            target = torch.arange(0, query_embeddings.size(0), dtype=torch.long, device=query_embeddings.device)
            loss = self.criterion(similarity, target)
            loss = loss.sum() / mask_sum  # Normalize loss by the number of unmasked elements
        else:
            target = torch.arange(0, query_embeddings.size(0), dtype=torch.long, device=query_embeddings.device)
            loss = self.criterion(similarity, target)

        return loss
