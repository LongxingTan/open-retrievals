import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


class FGM:
    """Fast Gradient Sign Attack"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.backup: Dict[str, torch.Tensor] = dict()

    def attack(self, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        """
        emb_name: model's embedding name
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name: str = "word_embeddings"):
        """
        emb_name: model's embedding name
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup.clear()


class EMA:
    """Exponential moving average

    Example:

        # init:
        ema = EMA(model, 0.999)
        ema.register()

        # train:
        def train():
            optimizer.step()
            ema.update()

        # eval:
        def evaluate():
            ema.apply_shadow()
            # evaluation code here
            ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = dict()
        self.backup = dict()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = dict()


class PGD:
    def __init__(self, model: nn.Module):
        self.model = model
        self.emb_backup = dict()
        self.grad_backup = dict()

    def attack(
        self, epsilon: float = 1.0, alpha: float = 0.3, emb_name: str = "word_embeddings", is_first_attack: bool = False
    ):
        """
        emb_name: set to your model's embedding name
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name: str = "word_embeddings"):
        """
        emb_name: set to your model's embedding name
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = dict()

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class AWP:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        adv_param: str = "weight",
        adv_lr: float = 0.0001,
        adv_eps: float = 0.001,
        adv_start_epoch: int = 0,
        adv_step: int = 1,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_start_epoch = adv_start_epoch
        self.adv_step = adv_step
        self.backup = dict()
        self.backup_eps = dict()
        self.scaler = scaler

    def attack_backward(self, inputs, criterion, labels, epoch: int):
        if self.adv_lr == 0 or epoch < self.adv_start_epoch:
            return

        logger.info('[AWP] start')
        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                y_preds = self.model(inputs)
                adv_loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
                # adv_loss, logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
                # adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(
        self,
    ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = dict()
        self.backup_eps = dict()
