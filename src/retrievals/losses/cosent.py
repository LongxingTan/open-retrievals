import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSentLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, embed1: torch.Tensor, embed2: torch.Tensor, labels=None):
        """
        https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py
        """
        pred_sims = F.cosine_similarity(embed1, embed2)
        # each position cosine of every pair
        pred_sims = pred_sims[:, None] - pred_sims[None, :]
        pred_sims = pred_sims - (1 - labels) * 1e12
        pred_sims = pred_sims.view(-1)
        # e^0 = 1, so it equals to add 1 in log
        pred_sims = torch.cat([torch.tensor([0.0], device=embed1.device), pred_sims], dim=0)
        return torch.logsumexp(pred_sims, dim=0)
