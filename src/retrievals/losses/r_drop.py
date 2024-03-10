"""
`R-Drop: Regularized Dropout for Neural Networks
<https://arxiv.org/abs/2106.14448>`_
"""

from torch import nn


class RDropLoss(nn.Module):
    def __init__(self, alpha=4, reduction="mean"):
        super(RDropLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self, inputs, labels):
        loss = self.entropy(inputs, labels.long())
        loss_kl = self.kl(inputs[::2], inputs[1::2]) + self.kl(inputs[1::2], inputs[::2])
        return loss + loss_kl / 4 * self.alpha
