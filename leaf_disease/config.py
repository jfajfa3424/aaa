"""Shared experiment constants and CLI helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CLASS_NAMES: list[str] = [
    "Healthy",
    "Early blight",
    "Late blight",
    "Leaf spot",
    "Powdery mildew",
    "Rust",
    "Mosaic",
    "Bacterial wilt",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    data_dir: str = "data/leaf"
    output_dir: str = "outputs"
    model: str = "resnet50_cbam"
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    img_size: int = 224
    pretrained: bool = False
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    patience: int = 20
    amp: bool = False
    demo: bool = False
    cifar10: bool = False
    demo_images_per_class: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def add_train_args(parser) -> None:
    parser.add_argument("--data-dir", default=TrainConfig.data_dir)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--model", default=TrainConfig.model)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--img-size", type=int, default=TrainConfig.img_size)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--optimizer", default=TrainConfig.optimizer, choices=["sgd", "rmsprop", "adam", "adamw"])
    parser.add_argument("--scheduler", default=TrainConfig.scheduler, choices=["cosine_warmup", "cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    parser.add_argument("--cutmix-alpha", type=float, default=TrainConfig.cutmix_alpha)
    parser.add_argument("--mixup-prob", type=float, default=TrainConfig.mixup_prob)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Train on a tiny synthetic 8-class set to verify the pipeline.")
    parser.add_argument("--cifar10", action="store_true", help="Use CIFAR-10 as a drop-in public dataset.")
    parser.add_argument("--demo-images-per-class", type=int, default=TrainConfig.demo_images_per_class)
