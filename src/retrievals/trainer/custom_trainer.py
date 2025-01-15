import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .adversarial import AWP, EMA, FGM

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """Training arguments dataclass for better configuration management"""

    epochs: int = 10
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = False
    logging_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 1000
    use_fgm: bool = False
    use_awp: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    use_swa: bool = False
    gradient_checkpointing: bool = False
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 4


class NLPTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize training components
        self.scaler = GradScaler(enabled=args.fp16)
        self.metrics_tracker = MetricsTracker()

        # Setup adversarial training if enabled
        self.setup_adversarial_training()

        # Move model to device
        self.model.to(self.device)

    def setup_adversarial_training(self):
        """Initialize adversarial training components"""
        if self.args.use_fgm:
            self.fgm = FGM(self.model)
        if self.args.use_awp:
            self.awp = AWP(self.model, self.optimizer)
        if self.args.use_ema:
            self.ema = EMA(self.model, self.args.ema_decay)
            self.ema.register()

    def train(self):
        """Main training loop"""
        global_step = 0

        for epoch in range(self.args.epochs):
            self.metrics_tracker = MetricsTracker()

            with tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}") as pbar:
                for step, batch in enumerate(pbar):
                    metrics = self.train_step(batch)
                    self.metrics_tracker.update(metrics, batch['labels'].size(0))

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        if self.args.use_ema:
                            self.ema.update()

                        if self.scheduler is not None:
                            self.scheduler.step()

                        global_step += 1

                    # Logging
                    if global_step % self.args.logging_steps == 0:
                        metrics = self.metrics_tracker.get_metrics()
                        pbar.set_postfix(**metrics)
                        logger.info(f"Step {global_step}: {metrics}")

                    # Evaluation
                    if global_step % self.args.evaluation_steps == 0:
                        eval_metrics = self.evaluate()
                        logger.info(f"Evaluation metrics: {eval_metrics}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluation loop"""
        self.model.eval()
        eval_metrics = MetricsTracker()

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs, batch['labels'])

            metrics = {'eval_loss': loss.item()}
            eval_metrics.update(metrics, batch['labels'].size(0))

        return eval_metrics.get_metrics()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast(enabled=self.args.fp16):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs, batch['labels'])
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        # Adversarial training
        if self.args.use_fgm:
            self.fgm.attack()
            with autocast(enabled=self.args.fp16):
                adv_outputs = self.model(**batch)
                adv_loss = adv_outputs.loss / self.args.gradient_accumulation_steps
            self.scaler.scale(adv_loss).backward()
            self.fgm.restore()

        metrics = {'loss': loss.item()}
        return metrics

    def save_model(self, path: str):
        """Save model state"""
        if self.args.use_ema:
            self.ema.apply_shadow()

        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'args': self.args,
        }
        torch.save(state_dict, path)

        if self.args.use_ema:
            self.ema.restore()

    def load_model(self, path: str):
        """Load model state"""
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if state_dict['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


class MetricsTracker:
    """Tracks multiple metrics during training"""

    def __init__(self):
        self.metrics = defaultdict(AverageMeter)

    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        for key, value in metrics_dict.items():
            self.metrics[key].update(value, batch_size)

    def get_metrics(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.metrics.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
