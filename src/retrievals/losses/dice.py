import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        # Ensure contiguous tensors
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        )

        # Apply reduction method
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class Dice(nn.Module):
    def __init__(self, feature_num):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.feature_num = feature_num
        self.bn = nn.BatchNorm1d(self.feature_num, affine=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_norm = self.bn(x)
        x_p = self.sigmoid(x_norm)
        return self.alpha * (1.0 - x_p) * x + x_p * x
