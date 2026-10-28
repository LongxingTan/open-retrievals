from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration"""

    # Teacher model
    teacher_model: Optional[str] = None
    teacher_checkpoint: Optional[str] = None
    teacher_model_type: str = "bert"

    # Distillation parameters
    temperature: float = 2.0
    alpha: float = 0.7

    # Loss types
    loss_types: List[str] = field(default_factory=lambda: ["logits"])
    kl_weight: float = 1.0
    mse_weight: float = 1.0
    mse_scale: float = 1.0

    # Layer mapping
    layer_mapping: Optional[List[int]] = None
    distill_layers: int = 5
