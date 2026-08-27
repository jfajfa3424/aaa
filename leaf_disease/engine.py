"""One-epoch train / eval loops, mixup/cutmix, and cosine-warmup schedule."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def mixup_cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets, targets, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        images, y_a, y_b, lam = _cutmix(images, targets, lam)
        return images, y_a, y_b, lam
    if torch.rand(1).item() > mixup_prob or mixup_alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def _cutmix(images: torch.Tensor, targets: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    n, _, h, w = images.shape
    index = torch.randperm(n, device=images.device)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = int(torch.randint(w, (1,)).item()), int(torch.randint(h, (1,)).item())
    x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    return mixed, targets, targets[index], lam


def cosine_warmup_lambda(epoch: int, epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(optimizer: Optimizer, name: str, epochs: int, warmup: int, step_size: int = 30):
    key = name.lower()
    if key == "none":
        return None
    if key == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if key == "cosine_warmup":
        fn = lambda epoch: cosine_warmup_lambda(epoch, epochs, warmup)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    raise ValueError(f"Unknown scheduler {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        n_correct += int((pred == targets).sum().item())
        n_seen += images.size(0)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return {
        "loss": total_loss / max(n_seen, 1),
        "acc": n_correct / max(n_seen, 1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp: bool = False,
    epoch: int = 0,
) -> dict:
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0
    iterator = tqdm(loader, desc=f"train {epoch + 1}", leave=False)
    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)
        images, y_a, y_b, lam = mixup_cutmix_batch(images, targets, mixup_alpha, cutmix_alpha, mixup_prob)
        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        n_correct += int((lam * (pred == y_a).float() + (1.0 - lam) * (pred == y_b).float()).sum().item())
        n_seen += images.size(0)
        iterator.set_postfix(loss=f"{loss.item():.3f}")
    return {"loss": total_loss / max(n_seen, 1), "acc": n_correct / max(n_seen, 1)}
