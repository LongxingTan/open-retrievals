"""Models package for Open-Retrievals"""

from .base import Base as BaseModel
from .embedding_auto import AutoModelForEmbedding
from .rerank import AutoModelForRanking, BaseRanker, ColBERT
from .retrieval_auto import AutoModelForRetrieval

__all__ = [
    "AutoModelForEmbedding",
    "AutoModelForRanking",
    "AutoModelForRetrieval",
    "BaseModel",
    "BaseRanker",
    "ColBERT",
]
