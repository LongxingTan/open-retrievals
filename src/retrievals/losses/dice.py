import torch
import torch.nn as nn


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
