import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSentLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_true, y_pred):
        # https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py
        y_true = y_true[::2, 0]
        y_true = (y_true[:, None] < y_true[None, :]).float()
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        y_pred = torch.cat([torch.tensor([0.0]), y_pred], dim=0)
        return torch.logsumexp(y_pred, dim=0)
