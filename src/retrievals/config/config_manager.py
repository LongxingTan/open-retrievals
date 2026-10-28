from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from .main import MainConfig


class ConfigManager:
    """Load and validate the canonical YAML schema."""

    @staticmethod
    def load_config(config_path: str, overrides: Optional[Union[Dict[str, Any], List[str]]] = None) -> MainConfig:
        """Load configuration from YAML file with optional overrides."""
        base_config = OmegaConf.load(config_path)

        # Apply command line overrides if provided
        if overrides:
            if isinstance(overrides, dict):
                override_config = OmegaConf.create(overrides)
            else:
                override_config = OmegaConf.from_dotlist(overrides)
            base_config = OmegaConf.merge(base_config, override_config)

        # Convert to structured config
        schema = OmegaConf.structured(MainConfig)
        config = OmegaConf.merge(schema, base_config)

        directory = Path(config_path).resolve().parent
        for key in ("train_file", "valid_file", "test_file"):
            value = config.data[key]
            if value and "://" not in value and not Path(value).is_absolute():
                config.data[key] = str(directory / value)
        if not Path(config.output_dir).is_absolute():
            config.output_dir = str(directory / config.output_dir)

        # Validate
        ConfigManager.validate_config(config)

        return config

    @staticmethod
    def load_config_from_dict(config_dict: Dict[str, Any]) -> MainConfig:
        """Load configuration from dictionary."""
        config = OmegaConf.create(config_dict)
        schema = OmegaConf.structured(MainConfig)
        config = OmegaConf.merge(schema, config)
        ConfigManager.validate_config(config)
        return config

    @staticmethod
    def validate_config(config: Union[DictConfig, MainConfig]) -> None:
        """Validate configuration values."""
        if isinstance(config, MainConfig):
            config = OmegaConf.structured(config)
        required_fields = ["model", "training", "data", "loss"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        valid_model_types = ["bert", "llm", "mixed"]
        if config.model.model_type not in valid_model_types:
            raise ValueError(f"Model type must be one of {valid_model_types}")

        if config.training.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError("training.mixed_precision must be no, fp16, or bf16")
        if config.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if config.data.negative_samples < 0:
            raise ValueError("data.negative_samples cannot be negative")
        if config.loss.temperature <= 0:
            raise ValueError("loss.temperature must be positive")
        if not 0 <= config.loss.label_smoothing < 1:
            raise ValueError("loss.label_smoothing must be in [0, 1)")

        valid_task_types = ["dense_retrieval", "colbert", "cross_encoder", "distillation"]
        if config.task_type not in valid_task_types:
            raise ValueError(f"Task type must be one of {valid_task_types}")

        if config.task_type == "distillation":
            raise ValueError("YAML distillation is not implemented")

        task_losses = {
            "dense_retrieval": {"multiple_negatives_ranking", "mnrl", "mnr", "infonce", "info_nce", "triplet"},
            "cross_encoder": {"bce"},
            "colbert": {"colbert"},
        }
        if config.loss.loss_type not in task_losses[config.task_type]:
            raise ValueError(f"Loss {config.loss.loss_type!r} is not supported by the {config.task_type} YAML workflow")
        if config.ranking is not None and config.ranking.objective != "classification":
            raise ValueError("The YAML reranking workflow currently supports classification only")
        if config.model.load_in_4bit or config.model.load_in_8bit:
            raise ValueError("Quantized YAML training is not implemented")
        if config.data.data_augmentation or config.data.hard_negative_path:
            raise ValueError("Prepare augmentation and hard negatives in the dataset before training")
        if config.model.projection_dim is not None:
            raise ValueError("Dense projection_dim is not implemented in the YAML workflow")
        if config.colbert is not None and config.colbert.mask_punctuation:
            raise ValueError("ColBERT punctuation masking is not implemented; set mask_punctuation=false")

    @staticmethod
    def save_config(config: Union[DictConfig, MainConfig], save_path: str) -> None:
        """Save configuration to YAML file."""
        if isinstance(config, MainConfig):
            config = OmegaConf.structured(config)
        OmegaConf.save(config, save_path)

    @staticmethod
    def print_config(config: Union[DictConfig, MainConfig]) -> None:
        """Pretty print configuration."""
        if isinstance(config, MainConfig):
            config = OmegaConf.structured(config)
        print(OmegaConf.to_yaml(config))

    @staticmethod
    def to_dict(config: Union[DictConfig, MainConfig]) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if isinstance(config, MainConfig):
            config = OmegaConf.structured(config)
        return OmegaConf.to_container(config, resolve=True)
