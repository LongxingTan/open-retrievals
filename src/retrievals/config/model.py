from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration for BERT and LLM models"""

    # Model selection
    model_type: str = "bert"  # bert, llm, mixed
    model_name_or_path: str = "bert-base-uncased"

    # Tokenizer settings
    tokenizer_name: Optional[str] = None
    use_fast_tokenizer: bool = True

    # Architecture settings
    pooling_method: str = "cls"  # cls, mean, max, last
    normalize_embeddings: bool = True

    # Sequence length
    max_length: Optional[int] = None

    # Model dimensions
    hidden_size: Optional[int] = None
    projection_dim: Optional[int] = None

    # ColBERT specific
    colbert_dim: int = 128

    # LoRA settings (for LLM)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[str] = None

    # Quantization settings (for LLM)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"

    # Cross-encoder settings
    is_cross_encoder: bool = False
