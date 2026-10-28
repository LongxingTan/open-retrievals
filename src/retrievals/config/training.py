from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration for different frameworks"""

    # Basic training parameters
    num_epochs: float = 3
    max_steps: int = -1
    use_cpu: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 32

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    warmup_steps: Optional[int] = None

    # Mixed precision and efficiency
    mixed_precision: str = "no"  # bf16, fp16, no
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False

    # Scheduler
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant, polynomial

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"  # no, steps, epoch
    save_steps: Optional[int] = None

    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Distributed training
    ddp_backend: Optional[str] = None
    ddp_find_unused_parameters: bool = False

    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=list)

    # Resume training
    resume_from_checkpoint: Optional[str] = None
