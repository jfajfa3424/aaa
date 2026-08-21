"""Folder dataset, CIFAR-10 adapter, and a color-coded synthetic demo set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import CIFAR10

from .config import CLASS_NAMES, TrainConfig, project_root
from .transforms import build_transforms


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the path, useful for error analysis."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def _paint_leaf(rng: np.random.RandomState, size: int, class_id: int) -> Image.Image:
    """Make a cheap but class-separable 'leaf' so TinyCNN can actually learn."""
    palettes = [
        (46, 160, 67),
        (196, 92, 28),
        (139, 69, 19),
        (90, 140, 40),
        (210, 210, 210),
        (180, 60, 40),
        (90, 40, 140),
        (50, 90, 40),
    ]
    bg = tuple(int(c + rng.randint(-12, 13)) for c in (34, 90, 40))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    color = palettes[class_id % len(palettes)]
    for _ in range(3 + class_id % 3):
        x0, y0 = rng.randint(4, size // 3), rng.randint(4, size // 3)
        x1, y1 = rng.randint(size // 2, size - 4), rng.randint(size // 2, size - 4)
        fill = tuple(int(np.clip(c + rng.randint(-25, 26), 0, 255)) for c in color)
        draw.ellipse([x0, y0, x1, y1], fill=fill)
    if class_id in {1, 2}:
        for _ in range(12):
            x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
            draw.ellipse([x, y, x + 3, y + 3], fill=(40, 20, 10))
    if class_id == 4:
        for y in range(0, size, 6):
            draw.line([(0, y), (size, y)], fill=(240, 240, 240), width=1)
    noise = rng.randint(0, 18, (size, size, 3), dtype=np.uint8)
    arr = np.clip(np.asarray(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_synthetic_dataset(root: Path, images_per_class: int = 48, size: int = 64, seed: int = 42) -> Path:
    rng = np.random.RandomState(seed)
    root = Path(root)
    splits = {
        "train": int(images_per_class * 0.70),
        "val": int(images_per_class * 0.15),
        "test": images_per_class - int(images_per_class * 0.70) - int(images_per_class * 0.15),
    }
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    for split, count in splits.items():
        for class_id, name in enumerate(CLASS_NAMES):
            folder = root / split / name.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(count):
                img = _paint_leaf(rng, size=size, class_id=class_id)
                img.save(folder / f"{split}_{class_id}_{k:03d}.png")
    (root / "classes.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return root


def resolve_data_dir(cfg: TrainConfig) -> tuple[Path, list[str]]:
    if cfg.demo:
        root = project_root() / "data" / "demo_leaf"
        build_synthetic_dataset(root, images_per_class=cfg.demo_images_per_class, size=max(64, cfg.img_size), seed=cfg.seed)
        return root, CLASS_NAMES
    if cfg.cifar10:
        return project_root() / "data" / "cifar10", list(CIFAR10.classes) if hasattr(CIFAR10, "classes") else [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    return Path(cfg.data_dir), CLASS_NAMES


def _cifar_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    root = project_root() / "data" / "cifar10"
    train_set = CIFAR10(root=str(root), train=True, download=True, transform=build_transforms(cfg.img_size, train=True))
    test_set = CIFAR10(root=str(root), train=False, download=True, transform=build_transforms(cfg.img_size, train=False))
    n_val = 5000
    n_train = len(train_set) - n_val
    generator = torch_generator(cfg.seed)
    train_split, val_split = __import__("torch").utils.data.random_split(
        train_set, [n_train, n_val], generator=generator
    )
    names = list(train_set.classes)
    return (
        _loader(train_split, cfg, shuffle=True),
        _loader(val_split, cfg, shuffle=False),
        _loader(test_set, cfg, shuffle=False),
        names,
    )


def torch_generator(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _loader(dataset: Dataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    if cfg.cifar10:
        return _cifar_loaders(cfg)

    root, class_names = resolve_data_dir(cfg)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {train_dir}. Put class folders under train/val/test, "
            "or run `python -m leaf_disease.preprocess`, or pass --demo / --cifar10."
        )
    train_ds = IndexedImageFolder(str(train_dir), transform=build_transforms(cfg.img_size, train=True))
    val_ds = IndexedImageFolder(str(val_dir), transform=build_transforms(cfg.img_size, train=False))
    test_path = test_dir if test_dir.is_dir() else val_dir
    test_ds = IndexedImageFolder(str(test_path), transform=build_transforms(cfg.img_size, train=False))
    names = list(train_ds.classes)
    return _loader(train_ds, cfg, True), _loader(val_ds, cfg, False), _loader(test_ds, cfg, False), names
