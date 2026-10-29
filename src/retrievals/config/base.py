from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from omegaconf import II, MISSING


@dataclass
class BaseConfig:
    """Base configuration class with common fields"""

    # Experiment settings
    exp_name: str = "experiment"
    seed: int = 42

    # Device settings
    device: str = "auto"
    gpu_num: int = 1

    # Logging and monitoring
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # Path settings
    output_dir: str = "./outputs"
    cache_dir: Optional[str] = None

    # Debug settings
    debug: bool = False
    max_samples: Optional[int] = None
