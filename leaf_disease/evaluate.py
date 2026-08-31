"""Evaluate a checkpoint: accuracy, per-class P/R/F1, confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .dataset import build_dataloaders
from .engine import evaluate
from .metrics import summarize
from .models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved leaf-disease classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/leaf")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--cifar10", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=int(ckpt.get("img_size", 224)),
        demo=args.demo,
        cifar10=args.cifar10,
    )
    _, val_loader, test_loader, class_names = build_dataloaders(cfg)
    names = ckpt.get("class_names", class_names)
    model = build_model(ckpt["model"], num_classes=len(names), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loader = test_loader if args.split == "test" else val_loader
    stats = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    report = summarize(np.array(stats["y_true"]), np.array(stats["y_pred"]), names)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
