from dataclasses import dataclass
from typing import Optional


@dataclass
class ColBERTConfig:
    """ColBERT specific configuration"""

    # ColBERT dimensions
    colbert_dim: int = 128

    # Masking and filtering
    mask_punctuation: bool = False
    similarity_metric: str = "cosine"  # cosine, l2

    # Indexing settings
    index_path: Optional[str] = None
    index_name: str = "colbert_index"
    nprobe: int = 10  # for FAISS

    # Centroid compression
    num_partitions: int = 100
    num_subvectors: int = 96

    # Query settings
    maxsim_alpha: float = 1.0  # scaling factor for maxsim
    centroid_scoring: bool = False
