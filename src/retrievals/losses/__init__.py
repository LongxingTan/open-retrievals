from torch import nn

from .arcface import ArcFaceAdaptiveMarginLoss
from .circle import MultiLabelCircleLoss
from .colbert_loss import ColbertLoss
from .cosent import CoSentLoss
from .cosine_similarity import CosineSimilarity
from .infonce import InfoNCE
from .pearson_loss import PearsonLoss
from .simcse import SimCSE
from .token_loss import TokenLoss
from .triplet import TripletLoss, TripletRankingLoss


class AutoLoss(nn.Module):
    def __init__(self, loss_name, loss_kwargs={}):
        super(AutoLoss, self).__init__()
        if loss_name == 'simcse':
            self.loss_fn = SimCSE(**loss_kwargs)
        elif loss_name == 'triplet':
            self.loss_fn = TripletLoss(**loss_kwargs)
        elif loss_name == 'infonce':
            self.loss_fn = InfoNCE(**loss_kwargs)
        elif loss_name == 'arcface':
            self.loss_fn = ArcFaceAdaptiveMarginLoss(**loss_kwargs)
        elif loss_name == 'cosent':
            self.loss_fn = CoSentLoss(**loss_kwargs)
        elif loss_name == 'pearson':
            self.loss_fn = PearsonLoss(**loss_kwargs)
        elif loss_name == 'triplet_ranking':
            self.loss_fn = TripletRankingLoss(**loss_kwargs)

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)
