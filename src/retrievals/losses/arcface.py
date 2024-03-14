"""
`ArcFace: Additive Angular Margin Loss for Deep Face Recognition
<https://arxiv.org/abs/1801.07698>`_
"""

import logging
import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class ArcFaceAdaptiveMarginLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        criterion: Union[nn.Module, Callable, None] = None,
        scale: float = 30.0,
        margin: float = 0.50,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = criterion
        self.scale = scale

        self.ls_eps = ls_eps
        self.arc_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.init_parameters()

        self.easy_margin = easy_margin
        self.set_margin(margin)

    def init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.arc_weight)

    def set_margin(self, margin: float):
        self.margin = margin
        self.cos_m = torch.from_numpy(np.asarray(np.cos(margin))).float()
        self.sin_m = torch.from_numpy(np.asarray(np.sin(margin))).float()
        self.th = nn.Parameter(torch.FloatTensor([math.cos(math.pi - margin)]), requires_grad=False)
        self.mm = nn.Parameter(
            torch.FloatTensor([math.sin(math.pi - margin) * margin]),
            requires_grad=False,
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, weight=None, margin: Optional[float] = None):
        if margin is not None:
            # dynamic margin
            self.set_margin(margin=margin)

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.arc_weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            one_hot = labels
        else:
            device = embeddings.device
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        output_dict = dict()
        output_dict["sentence_embedding"] = output
        if self.criterion:
            output_dict["loss"] = self.criterion(output, one_hot, weight)
        return output_dict
