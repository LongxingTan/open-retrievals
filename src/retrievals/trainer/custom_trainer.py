import gc
import logging
import math
import time
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm

from .adversarial import AWP, EMA, FGM

logger = logging.getLogger(__name__)


def train_fn(
    epoch: int,
    model: nn.Module,
    train_loader,
    optimizer,
    criterion: Optional[Callable] = None,
    apex: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1,
    data_collator=None,
    batch_scheduler=None,
    fgm=None,
    awp=None,
    ema_inst=None,
    print_freq: int = 100,
    wandb: bool = False,
    device: str = "cuda",
    teacher: Optional[nn.Module] = None,
    teacher_loss: Optional[Callable] = None,
    **kwargs,
):
    start = time.time()
    global_step = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=apex)
    losses = AverageMeter()
    save_step = int(len(train_loader) / 1)

    # loss_list = []
    # metrics_list = []

    for step, inputs in enumerate(train_loader):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            inputs, labels = inputs
        elif isinstance(inputs, dict):
            # assert data_collator is not None, "if labels are in inputs keys, can't set data_collator to None"
            labels = inputs['labels']
            inputs.pop('labels', None)

        if data_collator:
            inputs = data_collator(inputs)

        if isinstance(inputs, (list, tuple)):
            for input in inputs:
                if isinstance(input, dict):
                    for k, v in input.items():
                        input[k] = v.to(device)
                else:
                    input = input.to(device)
        else:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=apex):
            if criterion is None:
                try:
                    preds = model(inputs, labels=labels)
                except ValueError:
                    preds = model(**inputs, labels=labels)
            else:
                try:
                    preds = model(inputs=inputs)
                except ValueError:
                    preds = model(**inputs)
            if isinstance(preds, dict) and "loss" in preds:
                loss = preds["loss"]
            else:
                if isinstance(inputs, dict):
                    if isinstance(preds, dict):
                        logits = preds['logits']
                    elif isinstance(preds, (list, tuple)):
                        logits = preds[0]
                    else:
                        logits = preds

                    if 'weights' in inputs:
                        loss = criterion(logits, labels.float(), inputs['weights'])
                    else:
                        loss = criterion(logits, labels.float())
                else:
                    loss = criterion(preds[0], preds[1])

        if teacher:
            with torch.no_grad():
                teacher_output = teacher(inputs)
                loss_distill = teacher_loss(preds, teacher_output)
                loss = loss_distill

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # if isinstance(preds, dict):
        #     acc = (preds["feature"].softmax(dim=1) * (labels > 0)).sum(axis=1).mean()
        # else:
        #     acc = (preds.softmax(dim=1) * (labels > 0)).sum(axis=1).mean()
        # acc = acc.detach().cpu().item()
        # acc_list.append(acc)

        # ---------------------fgm-------------
        if fgm:
            # attach the embedding
            fgm.attack(epsilon=1.0)
            with torch.cuda.amp.autocast(enabled=apex):
                preds = model(inputs)
                if isinstance(preds, dict) and "loss" in preds:
                    loss_avg = preds["loss"]
                else:
                    loss_avg = criterion(preds, labels)
            if gradient_accumulation_steps > 1:
                loss_avg = loss_avg / gradient_accumulation_steps
            losses.update(loss_avg.item(), batch_size)
            scaler.scale(loss_avg).backward()
            # restore the embedding
            fgm.restore()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            if ema_inst:
                ema_inst.update()

            optimizer.zero_grad()
            global_step += 1
            if batch_scheduler:
                batch_scheduler.step()

        if step % print_freq == 0 or (step + 1) == len(train_loader):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  "
                # "ACC: {acc: .4f}"
                .format(
                    epoch + 1,
                    step,
                    len(train_loader) - 1,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=batch_scheduler.get_lr()[0] if batch_scheduler else optimizer.param_groups[0]["lr"],
                    # acc=np.mean(acc_list),
                )
            )

        if wandb and step % print_freq == 0:
            print({"[loss": losses.val, "[lr": optimizer.param_groups[0]["lr"]})

        if (step + 1) % save_step == 0 and epoch > -1:
            if ema_inst:
                ema_inst.apply_shadow()

            if ema_inst:
                ema_inst.restore()

    return losses.avg


@torch.no_grad()
def valid_fn(
    epoch,
    valid_loader,
    model,
    criterion,
    eval_metrics=None,
    apex: bool = False,
    device="cuda",
    data_collator=None,
    print_freq: int = 100,
):
    start = time.time()
    model.eval()
    losses = AverageMeter()
    pred_array = torch.tensor([], device=device)
    label_array = np.array([])

    for step, inputs in enumerate(valid_loader):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            inputs, labels = inputs
        elif isinstance(inputs, dict):
            labels = inputs['labels']
            inputs.pop('labels', None)

        if data_collator:
            inputs = data_collator(inputs)

        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=apex):
            try:
                preds = model(inputs, labels=labels)
            except ValueError:
                preds = model(inputs)
            else:
                logger.warning('inputs of model should be with or without labels')

            if isinstance(preds, dict) and "loss" in preds:
                loss = preds["loss"]
                pred = preds["output"]
            else:
                loss = criterion(preds, labels)
                pred = preds

        losses.update(loss.item(), batch_size)
        pred_array = torch.cat((pred_array, pred), dim=0)
        label_array = np.append(label_array, np.array(labels))

        if step % print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                # "ACC: {acc: .4f}"
                .format(
                    epoch + 1,
                    step,
                    len(valid_loader),
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                    loss=losses,
                    # acc=np.mean(acc_list),
                )
            )

    if eval_metrics is not None:
        score = eval_metrics(label_array, pred_array)
        print(f"Epoch {epoch}, Score: {score}")
        return losses.avg, score

    return losses.avg


def inference_fn(test_loader, model, device):
    preds: List[np.ndarray] = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to("cpu").numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


class CustomTrainer(object):
    def __init__(self, model: Union[nn.Module], device: Optional[str] = None, apex: bool = False, teacher=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
        self.apex = apex
        self.train_fn = train_fn
        self.valid_fn = valid_fn
        if teacher:
            self.teacher = teacher.eval()
            self.teacher_loss = torch.nn.MSELoss()

    def train(
        self,
        train_loader,
        optimizer,
        epochs=10,
        criterion=None,
        scheduler=None,
        valid_loader=None,
        eval_metrics=None,
        data_collator=None,
        max_grad_norm=10,
        use_fgm: bool = False,
        use_awp: bool = False,
        ema_decay: float = 0,
        dynamic_margin_fn: Optional[Callable] = None,
        gradient_checkpointing: bool = False,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        logger.info('-------START TO TRAIN-------')
        if resume_from_checkpoint is not None:
            checkpoint = torch.load(resume_from_checkpoint)
            self.model.load_state_dict(checkpoint)

        if use_fgm:
            logger.info('[FGM] Use FGM adversarial')
            fgm = FGM(self.model.to(self.device))
        elif use_awp:
            logger.info('[AWP] Use AWP adversarial')
            awp = AWP(self.model.to(self.device), optimizer)

        if ema_decay > 0:
            logger.info('[EMA] Use EMA while training')
            ema_inst = EMA(self.model.to(self.device), ema_decay)
            ema_inst.register()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        for epoch in range(epochs):
            if dynamic_margin_fn:
                margin = dynamic_margin_fn(epoch)
                logger.info(f"[Dynamic margin] Epoch: {epoch}, Margin: {margin}")
                if criterion:
                    criterion.set_margin(margin)
                elif self.model.loss_fn is not None:
                    self.model.loss_fn.set_margin(margin)
                else:
                    raise ValueError(
                        'dynamic margin is True, but criterion of trainer and loss_fn of model are both none'
                    )

            # lr = scheduler(optimizer, epoch)

            # 放在这里, 是因为前面的set_margin也会产生新的参数位于cpu
            # 如果把参数直接放在cuda, 则导致parameter里没有出现该参数，无法分别设置学习率
            self.model = self.model.to(self.device)
            optimizer.zero_grad()
            self.train_fn(
                epoch=epoch,
                train_loader=train_loader,
                model=self.model,
                criterion=criterion,
                optimizer=optimizer,
                batch_scheduler=scheduler,
                fgm=fgm if use_fgm else None,
                awp=awp if use_awp else None,
                ema_inst=ema_inst if ema_decay > 0 else None,
                apex=self.apex,
                gradient_accumulation_steps=1,
                max_grad_norm=max_grad_norm,
                valid_loader=valid_loader,
                data_collator=data_collator,
                device=self.device,
            )

            if valid_loader is not None:
                self.valid_fn(
                    epoch=epoch,
                    valid_loader=valid_loader,
                    model=self.model,
                    criterion=criterion,
                    eval_metrics=eval_metrics,
                    apex=self.apex,
                    device=self.device,
                    data_collator=None,
                )

        torch.cuda.empty_cache()
        gc.collect()

    def predict(self, test_loader):
        self.model.eval()
        tbar = tqdm(test_loader)

        vectors = []
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(tbar):
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                vec = self.model(inputs)
                if isinstance(vec, dict):
                    vec = vec["embeddings"]
                vectors.append(vec.detach().cpu().numpy())

        V = np.concatenate(vectors)
        return V

    def save_state(self, output_dir: Optional[str] = None):
        # TODO: remove the loss_fn while save weights
        if not output_dir:
            output_dir = "./model.pth"
        torch.save(self.model.state_dict(), output_dir)

    def save_model(self, output_dir: Optional[str] = None, state_dict=None):
        pass


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


def asMinutes(s):
    m = math.floor(s / 60)
    s = s - m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
