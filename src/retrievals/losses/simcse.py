"""
`SimCSE: Simple Contrastive Learning of Sentence Embeddings
<https://arxiv.org/abs/2104.08821>`_
"""

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class SimCSE(nn.Module):
    """Cosine similarity = normalize + inner product"""

    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature: float = 0.05,
        dynamic_temperature: bool = False,
    ):
        super().__init__()
        self.criterion = criterion
        self.temperature = temperature
        if dynamic_temperature:
            # TODO
            self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: Optional[torch.Tensor] = None,
    ):
        similarity = F.cosine_similarity(query_embeddings.unsqueeze(1), pos_embeddings.unsqueeze(0), dim=-1)

        if neg_embeddings is not None:
            neg_similarity = F.cosine_similarity(query_embeddings.unsqueeze(1), neg_embeddings.unsqueeze(0), dim=-1)
            similarity = torch.cat([similarity, neg_similarity], dim=1)

        similarity = similarity / self.temperature

        target = torch.arange(0, query_embeddings.size(0), dtype=torch.long, device=query_embeddings.device)

        loss = self.criterion(similarity, target)
        return loss
