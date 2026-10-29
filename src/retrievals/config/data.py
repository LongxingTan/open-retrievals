from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataConfig:
    """Data configuration for datasets and preprocessing"""

    # Dataset paths
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    test_file: Optional[str] = None

    # Data format settings
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    data_format: str = "json"  # json, csv, parquet, huggingface

    # Column mappings
    query_col: str = "query"
    doc_col: str = "document"
    positive_col: str = "positive"
    negative_col: str = "negative"
    label_col: str = "label"
    score_col: str = "score"

    # Data processing
    max_length: int = 512
    query_max_length: int = 128
    doc_max_length: int = 512

    # Negative sampling
    negative_samples: int = 8
    use_inbatch_negative: bool = True
    hard_negative_path: Optional[str] = None

    # Validation settings
    valid_split: float = 0.1
    valid_max_samples: Optional[int] = None
    valid_query_file: Optional[str] = None
    valid_corpus_file: Optional[str] = None

    # Augmentation
    data_augmentation: bool = False
    augmentation_config: Optional[Dict[str, Any]] = None
