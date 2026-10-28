from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LossConfig:
    """Loss function configuration"""

    # Loss type
    loss_type: str = "multiple_negatives_ranking"  # mnrl, triplet, colbert, margin_mse, distillation

    # Contrastive loss parameters
    temperature: float = 0.05

    # Triplet loss parameters
    margin: float = 0.5
    distance_metric: str = "cosine"  # cosine, euclidean

    # Ranking loss parameters
    ranking_loss_type: str = "margin_mse"  # margin_mse, pair_kl

    # ColBERT loss parameters
    colbert_loss_type: str = "colbert"  # colbert, token
    use_inbatch_negative: bool = True

    # Distillation parameters
    distillation_type: str = "logits"  # logits, hidden_states, attention
    distill_temperature: float = 2.0
    alpha: float = 0.7
    teacher_model: Optional[str] = None
    teacher_checkpoint: Optional[str] = None
    loss_types: List[str] = field(default_factory=lambda: ["logits"])

    # ArcFace and Circle loss parameters
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3

    # Multiple loss weights
    loss_weights: Optional[Dict[str, float]] = None

    # Label smoothing
    label_smoothing: float = 0.0

    # Special loss parameters
    r_dropout_alpha: float = 0.0
    focal_loss_alpha: float = 1.0
    focal_loss_gamma: float = 2.0
