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


def get_loss_fn(*args, **kwargs):
    from .factory import get_loss_fn as _get_loss_fn

    return _get_loss_fn(*args, **kwargs)


__all__ = [
    "ArcFaceAdaptiveMarginLoss",
    "ColbertLoss",
    "CoSentLoss",
    "CosineSimilarity",
    "InfoNCE",
    "MultiLabelCircleLoss",
    "PearsonLoss",
    "SimCSE",
    "TokenLoss",
    "TripletLoss",
    "TripletRankingLoss",
    "get_loss_fn",
]
