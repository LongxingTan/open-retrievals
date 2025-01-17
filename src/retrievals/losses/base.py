"""Base loss"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn


class Base(ABC, nn.Module):
    def __init__(self, negatives_cross_device: bool = False):
        super().__init__()
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Cannot do negatives_cross_device without distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """negatives cross device"""
        if t is None:
            return None

        try:
            return self._dist_gather_tensor1(t)
        except RuntimeError:
            return self._dist_gather_tensor2(t)

    def _dist_gather_tensor1(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """negatives cross device"""
        if t is None:
            return
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def _dist_gather_tensor2(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return
        all_tensors = dist_nn.all_gather(t)
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
