"""Split a class-folder dataset into train / val / test and drop unreadable files."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def stratified_split(paths: list[Path], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def preprocess(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    copy: bool = True,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    class_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {input_dir}")

    summary = {"classes": {}, "skipped": []}
    for class_dir in class_dirs:
        readable = []
        for path in iter_images(class_dir):
            if is_readable(path):
                readable.append(path)
            else:
                summary["skipped"].append(str(path))
        splits = stratified_split(readable, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        summary["classes"][class_dir.name] = {k: len(v) for k, v in splits.items()}
        for split, files in splits.items():
            dest_dir = output_dir / split / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dest = dest_dir / src.name
                if copy:
                    shutil.copy2(src, dest)
                else:
                    if dest.exists():
                        dest.unlink()
                    dest.symlink_to(src.resolve())
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split a class-folder image dataset.")
    parser.add_argument("--input-dir", required=True, help="Root with one folder per class.")
    parser.add_argument("--output-dir", default="data/leaf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--symlink", action="store_true", help="Symlink files instead of copying.")
    args = parser.parse_args()
    summary = preprocess(
        Path(args.input_dir),
        Path(args.output_dir),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        copy=not args.symlink,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
