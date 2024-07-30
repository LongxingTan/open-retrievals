import torch
from torch import distributed as dist
from torch import nn


class DistributedLoss(nn.Module):
    def __init__(self, criterion):
        super(DistributedLoss, self).__init__()
        self.criterion = criterion
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = self.criterion(dist_x, dist_y)
        return loss

    def gather_tensor(self, t: torch.Tensor):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
