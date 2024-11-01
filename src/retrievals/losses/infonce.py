"""
`Representation Learning with Contrastive Predictive Coding
<https://arxiv.org/abs/1807.03748v2>`_
"""

import logging
from typing import Callable, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCE(nn.Module):
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
        super().__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative
        self.negatives_cross_device = negatives_cross_device
        self.negative_mode = negative_mode
        if self.temperature > 0.5:
            logger.error('InfoNCE loss use normalized and inner product by default, temperature should be 0.01 ~ 0.1')
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Cannot do negatives_cross_device without distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ):
        if self.negatives_cross_device and self.use_inbatch_negative:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            if negative_embeddings is not None:
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

            return self.criterion(similarity, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
