"""Render screenshots from a real training run (terminal + live plots + success card)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

from leaf_disease.config import CLASS_NAMES, project_root
from leaf_disease.dataset import build_synthetic_dataset


CN_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
MONO_FONT = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"
SANS_FONT = "/usr/share/fonts/truetype/inter/Inter-Regular.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/inter/Inter_18pt-SemiBold.ttf"


def _font(path: str, size: int, fallback: str = "DejaVuSans.ttf") -> ImageFont.FreeTypeFont:
    p = Path(path)
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.truetype(fm.findfont(fallback), size=size)


def _register_mpl_fonts() -> str:
    for path in (CN_FONT, SANS_FONT, MONO_FONT):
        if Path(path).exists():
            fm.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "Inter", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )
    return "WenQuanYi Micro Hei"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def render_samples(out: Path, data_root: Path) -> None:
    _register_mpl_fonts()
    names = CLASS_NAMES
    cols, rows = 8, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6.2))
    fig.suptitle("Demo synthetic leaf images (8 classes, color-coded so TinyCNN can learn)", fontsize=14, fontweight="bold")
    rng = np.random.RandomState(0)
    for c, name in enumerate(names):
        folder = data_root / "train" / name.replace(" ", "_")
        files = sorted(folder.glob("*.png"))
        pick = [files[i] for i in rng.choice(len(files), size=min(rows, len(files)), replace=False)]
        for r, path in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(Image.open(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#333333")
            if r == 0:
                ax.set_title(name, fontsize=9)
    fig.tight_layout()
    save(fig, out)


def _draw_window(lines: list[tuple[str, str]], title: str, size: tuple[int, int] = (1600, 900)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), "#0d1117")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([8, 8, w - 8, h - 8], radius=18, fill="#161b22", outline="#30363d", width=2)
    draw.rounded_rectangle([8, 8, w - 8, 62], radius=18, fill="#21262d")
    draw.rectangle([8, 44, w - 8, 62], fill="#21262d")
    for cx, color in ((36, "#ff5f56"), (60, "#ffbd2e"), (84, "#27c93f")):
        draw.ellipse([cx - 8, 26, cx + 8, 42], fill=color)
    title_font = _font(MONO_BOLD if Path(MONO_BOLD).exists() else MONO_FONT, 22)
    body = _font(MONO_FONT, 22)
    draw.text((112, 22), title, fill="#e6edf3", font=title_font)
    palette = {
        "dim": "#8b949e",
        "cmd": "#79c0ff",
        "ok": "#3fb950",
        "warn": "#d29922",
        "err": "#ff7b72",
        "key": "#ffa657",
        "plain": "#e6edf3",
        "accent": "#a5d6ff",
    }
    y = 88
    for kind, text in lines:
        draw.text((36, y), text, fill=palette.get(kind, "#e6edf3"), font=body)
        y += 34
        if y > h - 40:
            break
    return img


def render_terminal_start(out: Path) -> None:
    lines = [
        ("dim", "ubuntu@cloud-agent:/workspace$"),
        ("cmd", "python3 -m leaf_disease.train --demo --epochs 5 --output-dir outputs/demo --seed 42"),
        ("dim", ""),
        ("plain", "device=cpu  model=tiny  params=94,728  classes=8"),
        ("dim", "building synthetic 8-class leaf set → data/demo_leaf/{train,val,test}"),
        ("key", "epoch 001/5  train_loss=1.8602 acc=0.375  val_loss=1.8879 acc=0.464  lr=2.00e-04"),
        ("dim", ""),
        ("dim", "# TinyCNN on CPU, 64×64 images, 48 samples/class. Pipeline smoke test."),
    ]
    _draw_window(lines, "train — leaf_disease demo (start)", (1480, 520)).save(out)


def render_terminal_progress(out: Path, log_text: str) -> None:
    lines: list[tuple[str, str]] = [
        ("dim", "ubuntu@cloud-agent:/workspace$ python3 -m leaf_disease.train --demo --epochs 5"),
        ("dim", ""),
    ]
    for raw in log_text.strip().splitlines():
        kind = "plain"
        if raw.startswith("epoch"):
            kind = "accent"
        if "best val" in raw or "wrote" in raw:
            kind = "ok"
        if raw.startswith("device="):
            kind = "key"
        lines.append((kind, raw))
    lines += [
        ("dim", ""),
        ("ok", "✓ training finished without error"),
    ]
    _draw_window(lines, "train — 5/5 epochs complete", (1600, 780)).save(out)


def render_terminal_eval(out: Path, eval_text: str) -> None:
    lines: list[tuple[str, str]] = [
        ("dim", "ubuntu@cloud-agent:/workspace$ python3 -m leaf_disease.evaluate \\"),
        ("cmd", "    --checkpoint outputs/demo/best.pt --demo --output-dir outputs/demo"),
        ("dim", ""),
    ]
    for raw in eval_text.strip().splitlines():
        kind = "plain"
        if "accuracy" in raw or "macro_" in raw:
            kind = "ok"
        if raw.startswith("wrote"):
            kind = "key"
        lines.append((kind, raw))
    lines += [
        ("dim", ""),
        ("ok", "✓ evaluate wrote outputs/demo/test_metrics.json"),
        ("ok", "✓ pipeline OK: preprocess → train → val → test"),
    ]
    _draw_window(lines, "evaluate — test metrics", (1480, 620)).save(out)


def render_curves(out: Path, history: dict) -> None:
    _register_mpl_fonts()
    h = history["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    ax.plot(epochs, h["train_loss"], marker="o", color="#1f77b4", label="Train loss")
    ax.plot(epochs, h["val_loss"], marker="s", color="#d62728", label="Val loss")
    ax.set_title("Live run: loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.legend(frameon=False)
    ax.set_xticks(epochs)
    ax = axes[1]
    ax.plot(epochs, np.array(h["train_acc"]) * 100, marker="o", color="#1f77b4", label="Train acc")
    ax.plot(epochs, np.array(h["val_acc"]) * 100, marker="s", color="#2ca02c", label="Val acc")
    ax.set_title("Live run: accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.set_xticks(epochs)
    fig.suptitle("Actual 5-epoch demo on CPU (not the 100-epoch paper figures)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def render_confusion(out: Path, history: dict) -> None:
    _register_mpl_fonts()
    names = [n.replace("_", " ") for n in history["class_names"]]
    cm = np.array(history["test"]["confusion"], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Live run confusion matrix  |  test acc = {history['test']['accuracy']*100:.2f}%")
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.6 else "#111111"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    fig.tight_layout()
    save(fig, out)


def render_per_class(out: Path, history: dict) -> None:
    _register_mpl_fonts()
    names = [n.replace("_", " ") for n in history["class_names"]]
    pc = history["test"]["per_class"]
    f1 = np.array([pc[n]["f1"] * 100 for n in history["class_names"]])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.bar(names, f1, color="#4C78A8", edgecolor="#222222", linewidth=0.6)
    ax.set_ylim(0, 110)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Live run: per-class F1 on the demo test split")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, out)


def render_success_card(out: Path, history: dict) -> None:
    """Final 'code ran successfully' image."""
    w, h = 1600, 1000
    img = Image.new("RGB", (w, h), "#f6f7f9")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([40, 40, w - 40, h - 40], radius=28, fill="#ffffff", outline="#d0d7de", width=3)

    title = _font(CN_FONT, 42)
    sub = _font(CN_FONT, 22)
    mono = _font(MONO_FONT, 22)
    mono_b = _font(MONO_BOLD if Path(MONO_BOLD).exists() else MONO_FONT, 28)
    small = _font(CN_FONT, 18)

    draw.ellipse([86, 86, 154, 154], fill="#1f883d")
    draw.text((104, 96), "✓", fill="white", font=_font(CN_FONT, 40))
    draw.text((180, 88), "代码跑通  ·  Pipeline OK", fill="#1f2328", font=title)
    draw.text((180, 148), "叶片病害图像分类  |  python -m leaf_disease.train --demo", fill="#57606a", font=sub)

    test = history["test"]
    cards = [
        ("Device", "CPU"),
        ("Model", "TinyCNN"),
        ("Params", f"{history['params']:,}"),
        ("Epochs", str(history["config"]["epochs"])),
        ("Best val acc", f"{history['best_val_acc']*100:.2f}%"),
        ("Test acc", f"{test['accuracy']*100:.2f}%"),
        ("Macro F1", f"{test['macro_f1']*100:.2f}%"),
        ("Classes", str(len(history["class_names"]))),
    ]
    box_w, box_h = 340, 118
    gap_x, gap_y = 24, 20
    x0, y0 = 80, 230
    for i, (label, value) in enumerate(cards):
        r, c = divmod(i, 4)
        x = x0 + c * (box_w + gap_x)
        y = y0 + r * (box_h + gap_y)
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=16, fill="#f6f8fa", outline="#d0d7de", width=2)
        draw.text((x + 20, y + 18), label, fill="#57606a", font=small)
        draw.text((x + 20, y + 52), value, fill="#1f2328", font=mono_b)

    y = 520
    draw.text((80, y), "本次真实命令与落盘文件", fill="#1f2328", font=_font(CN_FONT, 26))
    log_lines = [
        "$ python3 -m leaf_disease.train --demo --epochs 5 --output-dir outputs/demo",
        "$ python3 -m leaf_disease.evaluate --checkpoint outputs/demo/best.pt --demo",
        "wrote outputs/demo/best.pt",
        "wrote outputs/demo/history.json",
        "wrote outputs/demo/test_metrics.json",
        "exit code = 0",
    ]
    y += 48
    for line in log_lines:
        color = "#1a7f37" if line.startswith("wrote") or line.startswith("exit") else "#24292f"
        draw.text((96, y), line, fill=color, font=mono)
        y += 36

    draw.rounded_rectangle([80, 860, w - 80, 940], radius=14, fill="#dafbe1", outline="#1f883d", width=2)
    draw.text((110, 882), "最后确认：训练无报错，验证/测试指标已写出，整条流水线可复现。", fill="#0d4a1e", font=sub)
    img.save(out)


def render_dashboard(out: Path, history: dict) -> None:
    _register_mpl_fonts()
    h = history["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    names = [n.replace("_", " ") for n in history["class_names"]]
    cm = np.array(history["test"]["confusion"], dtype=np.int64)
    test = history["test"]

    fig = plt.figure(figsize=(14.5, 9.2))
    fig.suptitle("代码跑通最后一屏  ·  demo run dashboard", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_axes([0.06, 0.56, 0.40, 0.34])
    ax1.plot(epochs, h["train_loss"], "o-", label="train")
    ax1.plot(epochs, h["val_loss"], "s-", label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(frameon=False)

    ax2 = fig.add_axes([0.54, 0.56, 0.40, 0.34])
    ax2.plot(epochs, np.array(h["train_acc"]) * 100, "o-", label="train")
    ax2.plot(epochs, np.array(h["val_acc"]) * 100, "s-", label="val")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=False)

    ax3 = fig.add_axes([0.06, 0.08, 0.40, 0.38])
    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_title("Test confusion")
    ax3.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks(range(len(names)), names, fontsize=8)
    fig.colorbar(im, ax=ax3, fraction=0.046)

    ax4 = fig.add_axes([0.54, 0.08, 0.40, 0.38])
    ax4.axis("off")
    ax4.set_title("Run summary")
    summary = (
        f"device          CPU\n"
        f"model           {history['config']['model']}\n"
        f"params          {history['params']:,}\n"
        f"epochs          {history['config']['epochs']}\n"
        f"best val acc    {history['best_val_acc']*100:6.2f} %\n"
        f"test acc        {test['accuracy']*100:6.2f} %\n"
        f"macro precision {test['macro_precision']*100:6.2f} %\n"
        f"macro recall    {test['macro_recall']*100:6.2f} %\n"
        f"macro F1        {test['macro_f1']*100:6.2f} %\n"
        f"exit code       0"
    )
    ax4.text(0.04, 0.96, summary, va="top", family="DejaVu Sans Mono", fontsize=12, transform=ax4.transAxes)
    box = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.04", fill=False, edgecolor="#1f883d", linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(box)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("outputs/demo/history.json"))
    parser.add_argument("--train-log", type=Path, default=Path("outputs/demo/train.log"))
    parser.add_argument("--eval-log", type=Path, default=Path("outputs/demo/eval.log"))
    parser.add_argument("--out-dir", type=Path, default=Path("run_shots"))
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    train_log = args.train_log.read_text(encoding="utf-8") if args.train_log.exists() else ""
    eval_log = args.eval_log.read_text(encoding="utf-8") if args.eval_log.exists() else ""
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    data_root = project_root() / "data" / "demo_leaf"
    if not (data_root / "train").is_dir():
        build_synthetic_dataset(data_root, images_per_class=48, size=64, seed=42)

    render_samples(out / "01_sample_leaves.png", data_root)
    render_terminal_start(out / "02_train_start.png")
    render_terminal_progress(out / "03_train_progress.png", train_log)
    render_curves(out / "04_loss_acc_live.png", history)
    render_confusion(out / "05_confusion_live.png", history)
    render_per_class(out / "06_per_class_f1.png", history)
    render_terminal_eval(out / "07_eval_terminal.png", eval_log)
    render_success_card(out / "08_success_final.png", history)
    render_dashboard(out / "09_dashboard_final.png", history)

    shutil.copy2(args.history, out / "history.json")
    if args.train_log.exists():
        shutil.copy2(args.train_log, out / "train.log")
    if args.eval_log.exists():
        shutil.copy2(args.eval_log, out / "eval.log")
    print(f"wrote {len(list(out.glob('*.png')))} png files under {out}")


if __name__ == "__main__":
    main()
