"""Loss construction for the supported YAML training contracts."""

from torch import nn

from .colbert_loss import ColbertLoss
from .infonce import InfoNCE
from .triplet import TripletLoss


def get_loss_fn(config):
    name = config.loss_type.lower()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if name in {"multiple_negatives_ranking", "mnrl", "mnr", "infonce", "info_nce"}:
        return InfoNCE(
            temperature=config.temperature,
            criterion=criterion,
            use_inbatch_negative=config.use_inbatch_negative,
        )
    if name == "triplet":
        return TripletLoss(margin=config.margin, temperature=config.temperature)
    if name == "colbert":
        return ColbertLoss(
            temperature=config.temperature,
            criterion=criterion,
            use_inbatch_negative=config.use_inbatch_negative,
        )
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Loss {name!r} has no YAML batch contract; instantiate its Python class directly")
