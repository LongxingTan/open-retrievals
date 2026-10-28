from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration"""

    # Retrieval mode
    mode: str = "dense"  # dense, sparse, hybrid

    # Dense retrieval
    dense_index_path: Optional[str] = None
    dense_top_k: int = 100
    normalize_embeddings: bool = True

    # Sparse retrieval
    sparse_index_path: Optional[str] = None
    sparse_top_k: int = 100
    sparse_weight: float = 0.5

    # Hybrid retrieval
    hybrid_fusion: str = "rrf"  # rrf (reciprocal rank fusion), linear
    rrf_k: int = 60

    # ANN settings
    ann_algorithm: str = "faiss"  # faiss, hnsw, scann
    ann_metric: str = "cosine"  # cosine, l2, ip

    # Faiss specific
    faiss_index_type: str = "IndexFlatIP"  # IndexFlatIP, IndexIVFFlat, IndexHNSW
    nlist: int = 100
    nprobe: int = 10

    # HNSW specific
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
