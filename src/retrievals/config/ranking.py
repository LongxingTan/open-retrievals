from dataclasses import dataclass
from typing import Optional


@dataclass
class RankingConfig:
    """Cross-encoder and reranking configuration"""

    # Ranking settings
    is_cross_encoder: bool = False
    rerank_top_k: int = 100
    max_rerank_depth: int = 1000

    # Objective type
    objective: str = "classification"  # classification, regression, listwise

    # Listwise ranking
    list_size: int = 20
    list_loss_type: str = "listnet"  # listnet, listmle, approxndcg

    # Scoring
    score_scale: float = 1.0
    score_bias: float = 0.0

    # Regularization
    length_normalization: bool = True
    query_length_penalty: float = 0.0
