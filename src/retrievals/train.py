"""Assemble explicit task trainers from the same configuration used by the CLI."""

import json
from inspect import signature
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, set_seed

from .config import ConfigManager, MainConfig
from .data.collator import ColBertCollator, RerankCollator
from .data.training import RetrievalDataset, load_training_dataset
from .data.training_collator import DenseCollator
from .losses import get_loss_fn
from .models import AutoModelForEmbedding, AutoModelForRanking, ColBERT
from .trainer.colbert import ColBERTTrainer
from .trainer.dense import DenseTrainer
from .trainer.rerank import CrossEncoderTrainer


def resolve_config(config, overrides=None):
    if isinstance(config, (str, Path)):
        return ConfigManager.load_config(str(config), overrides)
    if overrides:
        raise ValueError("Pass overrides with a YAML path, or update the Python configuration directly")
    if isinstance(config, MainConfig):
        config = OmegaConf.structured(config)
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    return ConfigManager.load_config_from_dict(config)


def load_model(config):
    """Load a backbone and task head on CPU; Trainer owns device placement."""
    settings = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        settings.tokenizer_name or settings.model_name_or_path, use_fast=settings.use_fast_tokenizer
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    loader = AutoModelForSequenceClassification if config.task_type == "cross_encoder" else AutoModel
    kwargs = {"num_labels": 1} if config.task_type == "cross_encoder" else {}
    adapter_path = Path(settings.model_name_or_path) / "adapter_config.json"
    if adapter_path.is_file():
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(settings.model_name_or_path)
        backbone = loader.from_pretrained(adapter_config.base_model_name_or_path, **kwargs)
        backbone = PeftModel.from_pretrained(backbone, settings.model_name_or_path, is_trainable=settings.use_lora)
    else:
        backbone = loader.from_pretrained(settings.model_name_or_path, **kwargs)
    if len(tokenizer) > backbone.get_input_embeddings().num_embeddings:
        backbone.resize_token_embeddings(len(tokenizer))
    backbone.config.pad_token_id = tokenizer.pad_token_id
    if settings.use_lora and not adapter_path.is_file():
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS if config.task_type == "cross_encoder" else TaskType.FEATURE_EXTRACTION,
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=settings.lora_target_modules.split(",") if settings.lora_target_modules else None,
        )
        backbone = get_peft_model(backbone, lora)
    if config.training.gradient_checkpointing:
        backbone.config.use_cache = False
    common = {"model": backbone, "tokenizer": tokenizer, "device": "cpu"}
    if config.task_type == "cross_encoder":
        return AutoModelForRanking(**common)
    if config.task_type == "colbert":
        dim = config.colbert.colbert_dim if config.colbert else settings.colbert_dim
        linear = torch.nn.Linear(backbone.config.hidden_size, dim)
        weights = Path(settings.model_name_or_path) / "colbert_linear.pt"
        if weights.is_file():
            linear.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return ColBERT(**common, linear_layer=linear)
    return AutoModelForEmbedding(
        **common,
        pooling_method=settings.pooling_method,
        max_length=config.data.doc_max_length,
        normalize_embeddings=settings.normalize_embeddings,
    )


def training_arguments(config, training=True):
    settings = config.training
    if training and settings.eval_strategy != "no" and not (config.data.valid_file or config.data.valid_query_file):
        raise ValueError("Set data.valid_file when training.eval_strategy enables evaluation")
    optimizers = {"adamw": "adamw_torch", "sgd": "sgd"}
    if settings.optimizer not in optimizers:
        raise ValueError("training.optimizer must be adamw or sgd")
    args = dict(
        output_dir=config.output_dir,
        num_train_epochs=settings.num_epochs,
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        learning_rate=settings.learning_rate,
        optim=optimizers[settings.optimizer],
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        warmup_steps=settings.warmup_steps or 0,
        lr_scheduler_type=settings.lr_scheduler_type,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_grad_norm=settings.max_grad_norm,
        gradient_checkpointing=settings.gradient_checkpointing,
        # Non-reentrant checkpointing preserves adapter gradients with frozen embeddings.
        gradient_checkpointing_kwargs={"use_reentrant": False} if settings.gradient_checkpointing else None,
        eval_steps=settings.eval_steps,
        save_strategy=settings.save_strategy,
        save_steps=settings.save_steps or 500,
        save_total_limit=settings.save_total_limit,
        load_best_model_at_end=settings.load_best_model_at_end if training else False,
        metric_for_best_model=settings.metric_for_best_model,
        greater_is_better=settings.greater_is_better,
        bf16=settings.mixed_precision == "bf16",
        fp16=settings.mixed_precision == "fp16",
        use_cpu=settings.use_cpu or config.device == "cpu",
        logging_steps=settings.logging_steps,
        report_to=list(settings.report_to),
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        ddp_backend=settings.ddp_backend,
        ddp_find_unused_parameters=settings.ddp_find_unused_parameters,
    )
    # This renamed HF argument is the only version-specific translation.
    key = "eval_strategy" if "eval_strategy" in signature(TrainingArguments).parameters else "evaluation_strategy"
    args[key] = settings.eval_strategy if training else "no"
    return TrainingArguments(**args)


def build_trainer(config, *, training=True):
    """Build one task. Evaluation does not load the training dataset."""
    config = resolve_config(config)
    args = training_arguments(config, training=training)
    set_seed(config.seed)
    train_dataset = load_training_dataset(config.data, config.task_type) if training else None
    eval_dataset = load_training_dataset(config.data, config.task_type, training=False)
    if not training and eval_dataset is None and not config.data.valid_query_file:
        raise ValueError("Evaluation requires data.valid_file")
    for dataset in (train_dataset, eval_dataset):
        if not isinstance(dataset, RetrievalDataset):
            continue
        counts = [len(dataset[index].get("negative", [])) for index in range(len(dataset))]
        requires_negatives = (
            config.task_type == "cross_encoder"
            or config.loss.loss_type == "triplet"
            or not config.loss.use_inbatch_negative
        )
        missing = [index for index, count in enumerate(counts) if count == 0]
        if requires_negatives and missing:
            raise ValueError(f"Explicit negatives are required at dataset rows {missing}")
        if config.task_type != "cross_encoder" and not config.loss.use_inbatch_negative and len(set(counts)) > 1:
            raise ValueError("Per-row negative training requires equal negative counts in all rows")
    if eval_dataset is None and config.data.valid_query_file:
        eval_dataset = [0]  # Trainer requires a dataset when evaluation is enabled.
    model = load_model(config)
    data = config.data
    if config.task_type == "dense_retrieval":
        trainer_class = DenseTrainer
        collator = DenseCollator(model.tokenizer, data.query_max_length, data.doc_max_length)
    elif config.task_type == "cross_encoder":
        trainer_class = CrossEncoderTrainer
        collator = RerankCollator(model.tokenizer, max_length=data.max_length)
    else:
        trainer_class = ColBERTTrainer
        collator = ColBertCollator(
            model.tokenizer, query_max_length=data.query_max_length, document_max_length=data.doc_max_length
        )
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        loss_fn=get_loss_fn(config.loss),
        **({"label_smoothing": config.loss.label_smoothing} if config.task_type == "cross_encoder" else {}),
    )
    if config.data.valid_query_file:
        from .trainer.retrieval_metrics import RetrievalEvaluator

        trainer.retrieval_evaluator = RetrievalEvaluator(config.data, config.task_type)
    trainer.model_accepts_loss_kwargs = False
    model.device = args.device
    return trainer


def save_model(trainer, config, output_dir=None):
    """Export reloadable model weights separately from resumable Trainer checkpoints."""
    path = Path(output_dir) if output_dir else Path(config.output_dir) / "model"
    if trainer.is_world_process_zero():
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.save_pretrained(str(path))
        contract = {
            "task": config.task_type,
            "pooling_method": config.model.pooling_method,
            "normalize_embeddings": config.task_type != "cross_encoder",
            "max_length": config.data.max_length if config.task_type == "cross_encoder" else config.data.doc_max_length,
            "query_max_length": config.data.query_max_length,
            "document_max_length": config.data.doc_max_length,
            "colbert_dim": model.linear.out_features if config.task_type == "colbert" else None,
        }
        (path / "embedding_config.json").write_text(json.dumps(contract, indent=2))
        saved = OmegaConf.create(ConfigManager.to_dict(config))
        saved.model.model_name_or_path = str(path.resolve())
        saved.model.tokenizer_name = None
        saved.training.resume_from_checkpoint = None
        # Dictionary configurations resolve local paths against the working directory.
        # Preserve those locations when writing YAML into the export directory.
        for key in ("train_file", "valid_file", "test_file"):
            value = saved.data[key]
            if value and "://" not in value:
                saved.data[key] = str(Path(value).resolve())
        saved.output_dir = str(Path(saved.output_dir).resolve())
        ConfigManager.save_config(saved, str(path / "config.yaml"))
    return path


def train(config, overrides=None):
    config = resolve_config(config, overrides)
    trainer = build_trainer(config)
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    save_model(trainer, config)
    return trainer


def evaluate(config):
    return build_trainer(config, training=False).evaluate()


def main():
    import sys

    from .cli import main as cli_main

    cli_main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
