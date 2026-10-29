from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from .base import BaseConfig
from .colbert import ColBERTConfig
from .data import DataConfig
from .distillation import DistillationConfig
from .loss import LossConfig
from .model import ModelConfig
from .ranking import RankingConfig
from .retrieval import RetrievalConfig
from .training import TrainingConfig


@dataclass
class MainConfig(BaseConfig):
    """Main configuration combining all component configs"""

    # Component configs
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    data: DataConfig = MISSING
    loss: LossConfig = MISSING

    # Optional configs
    colbert: Optional[ColBERTConfig] = None
    ranking: Optional[RankingConfig] = None
    retrieval: Optional[RetrievalConfig] = None
    distillation: Optional[DistillationConfig] = None

    # Task type
    task_type: str = "dense_retrieval"  # dense_retrieval, colbert, cross_encoder, distillation
