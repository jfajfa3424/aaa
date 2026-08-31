"""Train VGG / ResNet / MobileNet / EfficientNet / ResNet-50+CBAM on leaf disease."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig, add_train_args
from .dataset import build_dataloaders
from .engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from .metrics import summarize
from .models import MODEL_NAMES, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for field_name in cfg.__dataclass_fields__:
        cli_name = field_name
        if hasattr(args, cli_name):
            cfg = replace(cfg, **{field_name: getattr(args, cli_name)})
    return cfg


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image classifier (leaf-disease domain).")
    add_train_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)

    if cfg.demo:
        cfg = replace(cfg, epochs=min(cfg.epochs, 8), img_size=min(cfg.img_size, 64), model="tiny", batch_size=min(cfg.batch_size, 16), mixup_alpha=0.0, cutmix_alpha=0.0, num_workers=0)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg)
    model = build_model(cfg.model, num_classes=len(class_names), pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.epochs, cfg.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type) if cfg.amp and device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_acc = -1.0
    stale = 0
    best_path = output_dir / "best.pt"

    print(f"device={device}  model={cfg.model}  params={count_parameters(model):,}  classes={len(class_names)}")
    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            mixup_prob=cfg.mixup_prob,
            scaler=scaler,
            amp=cfg.amp and device.type == "cuda",
            epoch=epoch,
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_acc"].append(val_stats["acc"])
        history["lr"].append(lr_now)
        print(
            f"epoch {epoch + 1:03d}/{cfg.epochs}  "
            f"train_loss={train_stats['loss']:.4f} acc={train_stats['acc']:.3f}  "
            f"val_loss={val_stats['loss']:.4f} acc={val_stats['acc']:.3f}  lr={lr_now:.2e}"
        )
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            stale = 0
            torch.save(
                {
                    "model": cfg.model,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": cfg.img_size,
                    "epoch": epoch + 1,
                    "val_acc": best_acc,
                    "config": cfg.to_dict(),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch + 1} (best val acc={best_acc:.3f})")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_stats = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    report = summarize(np.array(test_stats["y_true"]), np.array(test_stats["y_pred"]), class_names)
    run = {
        "config": cfg.to_dict(),
        "class_names": class_names,
        "params": count_parameters(model),
        "best_val_acc": best_acc,
        "history": history,
        "test": report,
        "models_supported": list(MODEL_NAMES),
    }
    save_json(output_dir / "history.json", run)
    save_json(output_dir / "test_metrics.json", report)
    print(f"best val acc={best_acc:.4f}  test acc={report['accuracy']:.4f}")
    print(f"wrote {best_path} and {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
