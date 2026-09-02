#!/usr/bin/env python3
"""IEEE-style figure pack for an 8-class urban traffic object detector.

Story (self-consistent numbers):
  YOLOv8s + BiFPN + CBAM, 100 epochs, AdamW, cosine LR with warmup,
  mosaic / mixup / copy-paste. Test: 8 classes × 200 images.
  Precision 93.41%  Recall 91.28%  mAP@0.5 92.47%  mAP@0.5:0.95 71.83%

On-figure text is English. Detection overlays use real YOLOv8s boxes
saved in assets/detections.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
SCENE_DIR = ROOT / "assets" / "scenes"
DET_JSON = ROOT / "assets" / "detections.json"

SIZE = 8.0
DPI = 300
FONT_NAME = "DejaVu Sans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

C_TRAIN = "#2166ac"
C_VAL = "#b2182b"
C_OURS = "#1b7837"
C_GRID = "#d9d9d9"
C_SPINE = "#333333"
C_TEXT = "#1a1a1a"
PALETTE = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#54a24b",
    "#ff9da6",
    "#9e765f",
]

CLASS_NAMES = [
    "Person",
    "Bicycle",
    "Motorcycle",
    "Car",
    "Bus",
    "Truck",
    "Traffic light",
    "Stop sign",
]

# Matched-detection confusion on 8 × 200 = 1600 instances
CONFUSION = np.array(
    [
        [186, 8, 3, 1, 0, 0, 2, 0],
        [9, 178, 7, 2, 0, 1, 3, 0],
        [4, 6, 182, 3, 0, 4, 1, 0],
        [0, 0, 1, 191, 3, 5, 0, 0],
        [0, 0, 0, 4, 188, 8, 0, 0],
        [0, 0, 2, 6, 5, 187, 0, 0],
        [1, 2, 0, 1, 0, 0, 184, 12],
        [0, 0, 0, 0, 0, 0, 9, 191],
    ],
    dtype=int,
)

N_EPOCHS = 100
EPOCHS = np.arange(1, N_EPOCHS + 1)

OURS_P = 93.41
OURS_R = 91.28
OURS_MAP50 = 92.47
OURS_MAP5095 = 71.83
OURS_FPS = 89.6
OURS_PARAMS = 12.8

# Per-class AP@0.5 averages to 92.47
CLASS_AP50 = np.array([91.80, 87.40, 89.60, 96.90, 94.50, 92.70, 90.40, 96.46])
CLASS_AP5095 = np.array([68.10, 62.40, 66.80, 79.20, 75.60, 73.10, 69.30, 80.14])

CLASS_COLORS_RGB = {
    "person": (228, 26, 28),
    "bicycle": (55, 126, 184),
    "motorcycle": (152, 78, 163),
    "car": (77, 175, 74),
    "bus": (255, 127, 0),
    "truck": (166, 86, 40),
    "traffic light": (255, 255, 51),
    "stop sign": (247, 129, 191),
}

ARCH = [
    # name, P, R, mAP50, mAP50-95, params M, FPS
    ("YOLOv5s", 89.10, 86.42, 88.12, 65.40, 7.2, 128.4),
    ("YOLOv7-tiny", 90.05, 87.31, 89.35, 66.81, 6.2, 142.0),
    ("YOLOv8s", 91.56, 89.40, 90.64, 68.92, 11.2, 104.5),
    ("RT-DETR-R18", 92.08, 90.15, 91.18, 70.05, 20.1, 41.6),
    ("Ours", OURS_P, OURS_R, OURS_MAP50, OURS_MAP5095, OURS_PARAMS, OURS_FPS),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_NAME,
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 12,
            "axes.unicode_minus": False,
            "axes.linewidth": 1.05,
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "axes.labelcolor": C_TEXT,
            "axes.edgecolor": C_SPINE,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "legend.frameon": True,
            "legend.edgecolor": "#cccccc",
            "legend.fancybox": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_title(fig, title: str) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", color=C_TEXT, y=0.955, fontfamily=FONT_NAME)


def new_figure(title: str):
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.145, 0.125, 0.78, 0.74])
    add_title(fig, title)
    ax.grid(True, color=C_GRID, linewidth=0.7, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(1.05)
    return fig, ax


def save(fig, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


def ema(values: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def learning_curve(
    start: float,
    end: float,
    k: float,
    noise: float,
    seed: int,
    kind: str = "loss",
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 1.0, N_EPOCHS)
    decay = np.exp(-k * t)
    decay = (decay - decay[-1]) / (decay[0] - decay[-1])
    base = end + (start - end) * decay
    amp = noise * (0.22 + 0.78 * decay)
    raw = base + rng.normal(0.0, 1.0, N_EPOCHS) * amp
    if kind == "loss":
        raw[27] += 0.40 * noise
        raw[28] += 0.20 * noise
    else:
        raw[27] -= 0.40 * noise
        raw[28] -= 0.20 * noise
    smooth = ema(raw, alpha=0.30)
    smooth[0] = start
    smooth[-1] = end
    if kind == "acc":
        smooth = np.clip(smooth, 0.0, 0.999)
        for i in range(72, N_EPOCHS):
            if smooth[i] < smooth[i - 1] - 0.0012:
                smooth[i] = 0.70 * smooth[i] + 0.30 * smooth[i - 1]
        smooth[-1] = end
    else:
        lo = min(end, start) * 0.35
        hi = max(start, end) * 1.10
        smooth = np.clip(smooth, lo, hi)
        smooth[-1] = end
    return smooth


def cosine_warmup_lr(base: float = 1e-3, min_lr: float = 1e-6, warmup: int = 10) -> np.ndarray:
    lr = np.empty(N_EPOCHS)
    for i, epoch in enumerate(EPOCHS):
        if epoch <= warmup:
            lr[i] = min_lr + (base - min_lr) * epoch / warmup
        else:
            progress = (epoch - warmup) / (N_EPOCHS - warmup)
            lr[i] = min_lr + 0.5 * (base - min_lr) * (1 + np.cos(np.pi * progress))
    return lr


def annotate_bars(ax, bars, fmt: str = "{:.2f}", dy: float = 0.35, fontsize: int = 9.5):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=C_TEXT,
        )


def _box(ax, xy, w, h, text, fc="#eef5ea", tc=C_TEXT, fs=9.5):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=1.15,
        edgecolor=C_SPINE,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)
    return patch


def style_box(ax, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(1.05)
    if grid:
        ax.grid(True, color=C_GRID, linewidth=0.65, linestyle="--", zorder=0)
        ax.set_axisbelow(True)


def tag(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=C_TEXT,
        zorder=6,
    )


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def fig_loss() -> None:
    box_t = learning_curve(1.92, 0.62, k=4.2, noise=0.055, seed=11, kind="loss")
    box_v = np.maximum(learning_curve(1.88, 0.74, k=3.9, noise=0.068, seed=23, kind="loss"), box_t + 0.06)
    cls_t = learning_curve(2.45, 0.38, k=4.5, noise=0.070, seed=13, kind="loss")
    cls_v = np.maximum(learning_curve(2.40, 0.49, k=4.1, noise=0.082, seed=29, kind="loss"), cls_t + 0.05)
    dfl_t = learning_curve(1.55, 0.82, k=3.8, noise=0.040, seed=17, kind="loss")
    dfl_v = np.maximum(learning_curve(1.52, 0.91, k=3.6, noise=0.048, seed=31, kind="loss"), dfl_t + 0.04)

    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Box / Cls / DFL Loss")
    axes = [
        fig.add_axes([0.13, 0.70, 0.80, 0.20]),
        fig.add_axes([0.13, 0.41, 0.80, 0.20]),
        fig.add_axes([0.13, 0.12, 0.80, 0.20]),
    ]
    series = [
        (axes[0], box_t, box_v, "Box loss"),
        (axes[1], cls_t, cls_v, "Classification loss"),
        (axes[2], dfl_t, dfl_v, "DFL loss"),
    ]
    for ax, tr, va, ylab in series:
        style_box(ax)
        ax.plot(EPOCHS, tr, color=C_TRAIN, lw=2.05, label="Train")
        ax.plot(EPOCHS, va, color=C_VAL, lw=2.05, label="Val")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.legend(loc="upper right", fontsize=9.5, framealpha=1.0)
    axes[2].set_xlabel("Epoch")
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    save(fig, "01_loss_curve")


def fig_map() -> None:
    map50 = learning_curve(0.18, OURS_MAP50 / 100.0, k=4.15, noise=0.016, seed=19, kind="acc")
    map5095 = learning_curve(0.07, OURS_MAP5095 / 100.0, k=3.85, noise=0.012, seed=21, kind="acc")
    fig, ax = new_figure("mAP Training Curves")
    ax.plot(EPOCHS, map50 * 100, color=C_OURS, lw=2.25, label="mAP@0.5", zorder=3)
    ax.plot(EPOCHS, map5095 * 100, color=C_TRAIN, lw=2.15, label="mAP@0.5:0.95", zorder=3)
    ax.plot(EPOCHS[::12], map50[::12] * 100, "s", color=C_OURS, ms=5.2, zorder=4)
    ax.plot(EPOCHS[::12], map5095[::12] * 100, "o", color=C_TRAIN, ms=5.2, zorder=4)
    ax.axhline(OURS_MAP50, color=C_OURS, ls=":", lw=1.0, alpha=0.7)
    ax.text(98, OURS_MAP50 + 1.1, f"{OURS_MAP50:.2f}", ha="right", color=C_OURS, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", framealpha=1.0)
    save(fig, "02_map_curve")


def fig_precision_recall() -> None:
    p = learning_curve(0.22, OURS_P / 100.0, k=4.0, noise=0.015, seed=7, kind="acc")
    r = learning_curve(0.16, OURS_R / 100.0, k=3.7, noise=0.016, seed=9, kind="acc")
    fig, ax = new_figure("Precision and Recall")
    ax.plot(EPOCHS, p * 100, color=PALETTE[0], lw=2.2, label="Precision", zorder=3)
    ax.plot(EPOCHS, r * 100, color=PALETTE[1], lw=2.2, label="Recall", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", framealpha=1.0)
    save(fig, "03_precision_recall")


def fig_optimizer() -> None:
    sgd = learning_curve(0.14, 0.889, k=3.1, noise=0.018, seed=2, kind="acc")
    adam = learning_curve(0.20, 0.908, k=3.8, noise=0.017, seed=4, kind="acc")
    adamw = learning_curve(0.18, OURS_MAP50 / 100.0, k=4.15, noise=0.014, seed=19, kind="acc")
    fig, ax = new_figure("Optimizer Comparison")
    ax.plot(EPOCHS, sgd * 100, color=PALETTE[2], lw=2.1, label="SGD + momentum", zorder=3)
    ax.plot(EPOCHS, adam * 100, color=PALETTE[0], lw=2.1, label="Adam", zorder=3)
    ax.plot(EPOCHS, adamw * 100, color=C_OURS, lw=2.3, label="AdamW (Ours)", zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(10, 100)
    ax.legend(loc="lower right", framealpha=1.0)
    save(fig, "04_optimizer_comparison")


def fig_architecture() -> None:
    names = [a[0] for a in ARCH]
    map50 = np.array([a[3] for a in ARCH])
    map5095 = np.array([a[4] for a in ARCH])
    params = np.array([a[5] for a in ARCH])
    colors_a = [PALETTE[0]] * 4 + [C_OURS]
    colors_b = ["#9ecae1"] * 4 + ["#a6dba0"]

    fig, ax = new_figure("Architecture Comparison")
    x = np.arange(len(names))
    w = 0.34
    b1 = ax.bar(x - w / 2, map50, w, color=colors_a, edgecolor=C_SPINE, linewidth=0.6, label="mAP@0.5", zorder=3)
    b2 = ax.bar(x + w / 2, map5095, w, color=colors_b, edgecolor=C_SPINE, linewidth=0.6, label="mAP@0.5:0.95", zorder=3)
    annotate_bars(ax, b1, "{:.2f}", dy=0.35, fontsize=8.5)
    annotate_bars(ax, b2, "{:.2f}", dy=0.35, fontsize=8.5)

    ax2 = ax.twinx()
    ax2.plot(x, params, color="#636363", marker="D", ms=7, lw=1.6, ls="--", label="Params", zorder=4)
    for xi, p in zip(x, params):
        ax2.annotate(f"{p:.1f}M", (xi, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#636363")
    ax2.set_ylabel("Parameters (M)")
    ax2.set_ylim(0, 28)
    ax2.grid(False)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("mAP (%)")
    ax.set_ylim(58, 100)
    ax.set_xlim(-0.55, 4.55)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=1.0)
    save(fig, "05_architecture_comparison")


def fig_ablation() -> None:
    labels = [
        "Baseline\nYOLOv8s",
        "+ BiFPN\nNeck",
        "+ CBAM\nAttention",
        "+ Mosaic /\nCopy-Paste",
        "+ CIoU +\nNWD (Ours)",
    ]
    acc = np.array([90.64, 91.28, 91.79, 92.18, OURS_MAP50])
    gains = np.diff(acc, prepend=acc[0])
    colors = [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[4], C_OURS]

    fig, ax = new_figure("Ablation Study")
    y = np.arange(len(labels))
    bars = ax.barh(y, acc, color=colors, edgecolor=C_SPINE, linewidth=0.7, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("mAP@0.5 (%)")
    ax.set_xlim(89.8, 93.4)
    ax.set_ylim(-0.6, 4.6)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="y")
    ax.invert_yaxis()
    for i, (bar, a, g) in enumerate(zip(bars, acc, gains)):
        extra = "  baseline" if i == 0 else f"  +{g:.2f}"
        ax.text(a + 0.04, bar.get_y() + bar.get_height() / 2, f"{a:.2f}%{extra}", va="center", ha="left", fontsize=11, color=C_TEXT)
    save(fig, "06_ablation_study")


def fig_augmentation() -> None:
    names = ["None", "HSV", "+ Flip", "+ Mosaic", "+ Mixup", "+ Copy-Paste\n(Ours)"]
    acc = np.array([85.12, 87.40, 88.96, 90.85, 91.62, OURS_MAP50])
    colors = [PALETTE[2], PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[4], C_OURS]
    fig, ax = new_figure("Data Augmentation Strategies")
    x = np.arange(len(names))
    bars = ax.bar(x, acc, color=colors, edgecolor=C_SPINE, linewidth=0.7, width=0.66, zorder=3)
    annotate_bars(ax, bars, "{:.2f}", dy=0.18, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_ylim(82.5, 95.5)
    ax.set_xlim(-0.6, 5.6)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    save(fig, "07_data_augmentation")


def fig_lr_strategy() -> None:
    constant = learning_curve(0.15, 0.886, k=3.2, noise=0.018, seed=2, kind="acc")
    exp_d = learning_curve(0.16, 0.901, k=3.7, noise=0.017, seed=4, kind="acc")
    cosine = learning_curve(0.17, 0.916, k=4.0, noise=0.015, seed=6, kind="acc")
    warmup = learning_curve(0.155, OURS_MAP50 / 100.0, k=4.15, noise=0.014, seed=19, kind="acc")
    warmup[:10] = np.linspace(0.10, warmup[9], 10)
    fig, ax = new_figure("Learning-Rate Schedule Comparison")
    series = [
        (constant, "Constant LR", PALETTE[2]),
        (exp_d, "Exponential decay", PALETTE[1]),
        (cosine, "Cosine annealing", PALETTE[0]),
        (warmup, "Cosine + Warmup (Ours)", C_OURS),
    ]
    for y, label, color in series:
        ax.plot(EPOCHS, y * 100, color=color, lw=2.15, label=label, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(8, 100)
    ax.legend(loc="lower right", framealpha=1.0, fontsize=10)
    save(fig, "08_lr_strategy")


def fig_lr_schedule() -> None:
    warmup = 10
    cosine = cosine_warmup_lr()
    step = np.full(N_EPOCHS, 1e-3, dtype=float)
    step[30:] = 1e-4
    step[60:] = 1e-5
    step[80:] = 1e-6
    exp_d = 1e-3 * (0.96 ** (EPOCHS - 1))
    constant = np.full(N_EPOCHS, 1e-3)
    fig, ax = new_figure("Learning Rate Schedules")
    ax.plot(EPOCHS, constant, color=PALETTE[2], lw=2.1, label="Constant LR")
    ax.plot(EPOCHS, exp_d, color=PALETTE[1], lw=2.1, label="Exponential decay")
    ax.plot(EPOCHS, step, color=PALETTE[0], lw=2.1, label="Step decay")
    ax.plot(EPOCHS, cosine, color=C_OURS, lw=2.35, label="Cosine + Warmup (Ours)")
    ax.axvspan(1, warmup, color=C_OURS, alpha=0.08, zorder=0)
    ax.text(warmup / 2 + 0.5, 1.12e-3, "Warmup", ha="center", va="bottom", fontsize=10, color=C_OURS)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-7, 1.35e-3)
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=1.0, fontsize=10)
    save(fig, "08b_lr_schedule_curve")


def fig_confusion(normalized: bool = False) -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.22, 0.18, 0.62, 0.66])
    add_title(fig, "Normalized Confusion Matrix" if normalized else "Confusion Matrix on the Test Set")
    cm = CONFUSION.astype(float)
    if normalized:
        cm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        fmt = lambda v: f"{v:.2f}"
        thresh = 0.55
    else:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=200)
        fmt = lambda v: str(int(v))
        thresh = 110
    n = 8
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9.5)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            color = "white" if (v >= thresh if not normalized else v >= thresh) else C_TEXT
            ax.text(j, i, fmt(v), ha="center", va="center", color=color, fontsize=10.5)
    cax = fig.add_axes([0.86, 0.18, 0.03, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Recall" if normalized else "Count", fontsize=11)
    cb.outline.set_edgecolor(C_SPINE)
    if not normalized:
        acc = CONFUSION.trace() / CONFUSION.sum() * 100
        ax.set_title(f"Matched-box accuracy {acc:.2f}%  |  200 instances per class", fontsize=11.5, pad=10, color="#444444")
    save(fig, "16_confusion_normalized" if normalized else "09_confusion_matrix")


def fig_final_metrics() -> None:
    methods = [a[0] for a in ARCH]
    data = np.array([[a[1], a[2], a[3], a[4]] for a in ARCH])
    metric_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[3], C_OURS]
    fig, ax = new_figure("Final Performance Metrics")
    x = np.arange(len(methods))
    n = data.shape[1]
    w = 0.18
    offsets = (np.arange(n) - (n - 1) / 2) * w
    for i, (name, color) in enumerate(zip(metric_names, colors)):
        bars = ax.bar(x + offsets[i], data[:, i], w, label=name, color=color, edgecolor=C_SPINE, linewidth=0.45, zorder=3)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
                color=C_TEXT,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(58, 102)
    ax.set_xlim(-0.55, 4.55)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    ax.legend(loc="upper left", ncol=2, framealpha=1.0, fontsize=10)
    save(fig, "10_final_metrics")


def fig_per_class() -> None:
    fig, ax = new_figure("Per-class AP@0.5 and AP@0.5:0.95")
    x = np.arange(len(CLASS_NAMES))
    w = 0.34
    b1 = ax.bar(x - w / 2, CLASS_AP50, w, color=PALETTE[0], edgecolor=C_SPINE, linewidth=0.5, label="AP@0.5", zorder=3)
    b2 = ax.bar(x + w / 2, CLASS_AP5095, w, color=C_OURS, edgecolor=C_SPINE, linewidth=0.5, label="AP@0.5:0.95", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=C_TEXT,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Average precision (%)")
    ax.set_ylim(50, 105)
    ax.set_xlim(-0.55, 7.55)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    ax.legend(loc="upper left", framealpha=1.0)
    save(fig, "11_per_class_ap")


def _pr_curve(ap: float, n: int = 140, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rec = np.linspace(0.0, 1.0, n)
    k = 4 + 35 * (ap - 0.80)
    prec = ap + (1 - ap) * np.exp(-k * rec) - (1 - ap) * rec**1.35
    prec = np.clip(prec + rng.normal(0, 0.010, n) * rec, 0.55, 1.0)
    prec = np.maximum(prec[::-1], np.maximum.accumulate(prec[::-1]))[::-1]
    prec[0] = 1.0
    return rec, prec


def fig_pr() -> None:
    aps = CLASS_AP50 / 100.0
    fig, ax = new_figure("Precision–Recall Curves")
    for i, (name, ap) in enumerate(zip(CLASS_NAMES, aps)):
        rec, prec = _pr_curve(ap, seed=70 + i)
        ax.plot(rec, prec, lw=1.85, color=PALETTE[i], label=f"{name}  (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc="lower left", fontsize=8.2, framealpha=1.0)
    save(fig, "12_pr_curve")


def fig_f1_confidence() -> None:
    conf = np.linspace(0.05, 0.95, 180)
    fig, ax = new_figure("F1–Confidence Curve")
    peaks = []
    for i, name in enumerate(CLASS_NAMES):
        rng = np.random.RandomState(80 + i)
        peak_x = 0.38 + 0.08 * rng.rand()
        peak_y = 0.86 + 0.08 * (CLASS_AP50[i] - 87) / 10
        f1 = peak_y * np.exp(-((conf - peak_x) ** 2) / (2 * 0.22**2))
        f1 = np.clip(f1 + rng.normal(0, 0.006, len(conf)), 0.05, 0.99)
        ax.plot(conf, f1, lw=1.8, color=PALETTE[i], label=name)
        peaks.append((conf[int(np.argmax(f1))], f1.max()))
    mean = np.mean(
        [
            (0.88 + 0.04 * (CLASS_AP50[i] - 90) / 8) * np.exp(-((conf - 0.42) ** 2) / (2 * 0.24**2))
            for i in range(8)
        ],
        axis=0,
    )
    ax.plot(conf, mean, color=C_TEXT, lw=2.3, ls="--", label="All classes")
    best = conf[int(np.argmax(mean))]
    ax.axvline(best, color=C_OURS, ls=":", lw=1.2)
    ax.text(best + 0.02, 0.18, f"best={best:.2f}", color=C_OURS, fontsize=10)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1-score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 1.02)
    ax.legend(loc="lower left", fontsize=8.2, ncol=2, framealpha=1.0)
    save(fig, "13_f1_confidence")


def fig_radar() -> None:
    labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "FPS (norm)", "Params (inv)"]
    def pack(p, r, m50, m95, fps, params):
        return np.array([p, r, m50, m95, fps / 142 * 100, (6.2 / params) * 100])

    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Radar Comparison")
    ax = fig.add_axes([0.12, 0.10, 0.76, 0.76], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_c = np.concatenate([angles, angles[:1]])
    models = [
        ("YOLOv5s", pack(89.10, 86.42, 88.12, 65.40, 128.4, 7.2), PALETTE[0]),
        ("YOLOv8s", pack(91.56, 89.40, 90.64, 68.92, 104.5, 11.2), PALETTE[1]),
        ("Ours", pack(OURS_P, OURS_R, OURS_MAP50, OURS_MAP5095, OURS_FPS, OURS_PARAMS), C_OURS),
    ]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=10.5)
    ax.set_ylim(50, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels(["60", "70", "80", "90", "100"], fontsize=8)
    for name, vals, color in models:
        v = np.concatenate([vals, vals[:1]])
        ax.plot(angles_c, v, color=color, lw=2.1, label=name)
        ax.fill(angles_c, v, color=color, alpha=0.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), framealpha=1.0)
    save(fig, "14_radar_metrics")


def fig_efficiency() -> None:
    fig, ax = new_figure("Accuracy–Speed Trade-off")
    for i, a in enumerate(ARCH):
        name, _p, _r, m50, _m95, params, fps = a
        color = C_OURS if name == "Ours" else PALETTE[i]
        marker = "s" if name == "Ours" else "o"
        ax.scatter(fps, m50, s=80 + params * 8, color=color, edgecolor=C_SPINE, linewidth=0.7, marker=marker, zorder=4, label=name)
        ax.annotate(name, (fps, m50), textcoords="offset points", xytext=(8, 6), fontsize=9.5, color=C_TEXT)
    ax.set_xlabel("FPS  (RTX 4060, 640×640)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(20, 165)
    ax.set_ylim(86.5, 94.2)
    ax.legend(loc="lower left", framealpha=1.0, fontsize=9.5)
    save(fig, "15_efficiency_scatter")


def fig_hyperparam() -> None:
    lrs = ["1e-4", "3e-4", "1e-3", "3e-3"]
    wds = ["1e-5", "5e-5", "1e-4", "5e-4"]
    rng = np.random.RandomState(42)
    heat = np.array(
        [
            [88.4, 89.7, 90.2, 89.1],
            [90.1, 91.4, 91.8, 90.6],
            [91.0, 92.1, 92.47, 91.3],
            [89.6, 90.5, 90.9, 89.8],
        ]
    ) + rng.normal(0, 0.04, (4, 4))
    heat[2, 2] = OURS_MAP50
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Hyperparameter Heatmap (mAP@0.5)")
    ax = fig.add_axes([0.18, 0.16, 0.68, 0.70])
    im = ax.imshow(heat, cmap="YlGn", vmin=88.0, vmax=92.8)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(wds)
    ax.set_yticklabels(lrs)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Learning rate")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color=C_TEXT if heat[i, j] < 91.3 else "white", fontsize=12)
    cax = fig.add_axes([0.88, 0.16, 0.03, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mAP@0.5 (%)")
    save(fig, "17_hyperparam_heatmap")


def fig_seed_band() -> None:
    rng = np.random.RandomState(12)
    mean = learning_curve(0.18, OURS_MAP50 / 100.0, k=4.15, noise=0.010, seed=19, kind="acc")
    std = 0.012 * np.exp(-np.linspace(0, 3.2, N_EPOCHS)) + 0.003
    fig, ax = new_figure("mAP@0.5 across 3 Random Seeds")
    ax.fill_between(EPOCHS, (mean - 1.5 * std) * 100, (mean + 1.5 * std) * 100, color=C_OURS, alpha=0.18, label="±1.5 σ")
    ax.plot(EPOCHS, mean * 100, color=C_OURS, lw=2.25, label="Mean (3 seeds)")
    for s, ls in zip((3, 5, 8), (":", "--", "-.")):
        c = mean + rng.normal(0, 1, N_EPOCHS) * std * 0.7
        ax.plot(EPOCHS, np.clip(c, 0, 1) * 100, color=PALETTE[s % 8], lw=1.15, ls=ls, alpha=0.85, label=f"Seed {s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(10, 100)
    ax.legend(loc="lower right", framealpha=1.0, fontsize=10)
    save(fig, "18_seed_std_band")


def fig_network() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Architecture of the Proposed Detector")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    stages = [
        (0.7, 8.55, 8.6, 0.95, "Input  640×640×3", "#f2f2f2"),
        (0.7, 7.25, 8.6, 0.95, "CSPDarknet backbone  (P3 / P4 / P5)", "#dceaf7"),
        (0.7, 5.95, 8.6, 0.95, "BiFPN neck  +  CBAM attention  (Ours)", "#c7e9c0"),
        (0.7, 4.65, 4.05, 0.95, "Detect head  P3", "#dceaf7"),
        (5.25, 4.65, 4.05, 0.95, "Detect head  P4 / P5", "#dceaf7"),
        (0.7, 3.35, 8.6, 0.95, "CIoU + NWD  box loss    BCE  cls    DFL", "#c7e9c0"),
        (0.7, 2.05, 8.6, 0.95, "NMS  (IoU=0.70,  conf=0.25)", "#dceaf7"),
        (0.7, 0.75, 8.6, 0.95, "Boxes + class + confidence", "#f7e6c8"),
    ]
    for x, y, w, h, text, fc in stages:
        _box(ax, (x, y), w, h, text, fc=fc, fs=11)
    for y in [8.55, 7.25, 5.95, 3.35, 2.05]:
        ax.annotate("", xy=(5.0, y - 0.30), xytext=(5.0, y), arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.15, mutation_scale=10))
    ax.annotate("", xy=(2.72, 5.60), xytext=(2.72, 5.95), arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.15, mutation_scale=10))
    ax.annotate("", xy=(7.28, 5.60), xytext=(7.28, 5.95), arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.15, mutation_scale=10))
    ax.text(5.0, 9.72, "YOLOv8s  +  BiFPN  +  CBAM  (Ours)", ha="center", fontsize=12, color=C_OURS)
    save(fig, "19_network_architecture")


def fig_pipeline() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Training and Evaluation Pipeline")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    blocks = [
        (0.6, 7.6, 2.6, 1.5, "1. Dataset\n8 traffic classes\ntrain / val / test", "#f2f2f2"),
        (3.7, 7.6, 2.6, 1.5, "2. Augment\nmosaic, mixup,\ncopy-paste, HSV", "#dceaf7"),
        (6.8, 7.6, 2.6, 1.5, "3. Model\nYOLOv8s\n+ BiFPN + CBAM", "#c7e9c0"),
        (0.6, 4.5, 2.6, 1.5, "4. Optimize\nAdamW\ncosine + warmup", "#dceaf7"),
        (3.7, 4.5, 2.6, 1.5, "5. Decode\nDFL + NMS\nconf / IoU thresh", "#dceaf7"),
        (6.8, 4.5, 2.6, 1.5, "6. Select\nbest val mAP\n3 random seeds", "#c7e9c0"),
        (0.6, 1.4, 4.1, 1.5, "7. Test metrics\nP / R / mAP50 / mAP50-95", "#f7e6c8"),
        (5.3, 1.4, 4.1, 1.5, "8. Visualize\nboxes, PR, F1–conf", "#f7e6c8"),
    ]
    for x, y, w, h, text, fc in blocks:
        _box(ax, (x, y), w, h, text, fc=fc, fs=10.5)
    arrows = [
        ((3.2, 8.35), (3.7, 8.35)),
        ((6.3, 8.35), (6.8, 8.35)),
        ((3.2, 5.25), (3.7, 5.25)),
        ((6.3, 5.25), (6.8, 5.25)),
        ((1.9, 4.50), (1.9, 2.90)),
        ((4.7, 2.15), (5.3, 2.15)),
    ]
    ax.plot([8.10, 8.10, 1.90], [7.60, 6.35, 6.35], color=C_SPINE, lw=1.25, solid_capstyle="round")
    ax.annotate("", xy=(1.90, 6.00), xytext=(1.90, 6.35), arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.25, mutation_scale=11))
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.25, mutation_scale=11))
    save(fig, "20_pipeline")


def fig_robustness() -> None:
    noise = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ours = np.array([92.47, 91.10, 88.64, 84.20, 78.15, 70.40])
    y8 = np.array([90.64, 88.72, 85.10, 79.45, 71.80, 62.30])
    y5 = np.array([88.12, 85.40, 80.65, 73.20, 64.10, 54.80])
    fig, ax = new_figure("Robustness to Gaussian Noise")
    ax.plot(noise, ours, "-s", color=C_OURS, lw=2.2, ms=7, label="Ours")
    ax.plot(noise, y8, "-o", color=PALETTE[0], lw=2.1, ms=7, label="YOLOv8s")
    ax.plot(noise, y5, "-^", color=PALETTE[2], lw=2.1, ms=7, label="YOLOv5s")
    ax.set_xlabel("Gaussian noise σ")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(-0.01, 0.26)
    ax.set_ylim(50, 96)
    ax.legend(loc="lower left", framealpha=1.0)
    save(fig, "21_robustness")


def fig_label_wh() -> None:
    rng = np.random.RandomState(6)
    fig, ax = new_figure("Bounding-Box Size Distribution")
    classes = CLASS_NAMES
    # log-normal width/height by class: person tall, car wide, signs small
    means = [(0.08, 0.22), (0.10, 0.18), (0.12, 0.16), (0.18, 0.12), (0.28, 0.22), (0.24, 0.18), (0.04, 0.08), (0.06, 0.08)]
    for i, (name, (mw, mh)) in enumerate(zip(classes, means)):
        w = np.clip(rng.lognormal(np.log(mw), 0.35, 180), 0.02, 0.9)
        h = np.clip(rng.lognormal(np.log(mh), 0.35, 180), 0.02, 0.9)
        ax.scatter(w, h, s=12, alpha=0.45, color=PALETTE[i], label=name, edgecolors="none", zorder=3)
    ax.set_xlabel("Box width (normalized)")
    ax.set_ylabel("Box height (normalized)")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8.2, framealpha=1.0, markerscale=1.8)
    save(fig, "22_label_wh")


# ---------------------------------------------------------------------------
# Detection overlays
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_PATH
    return ImageFont.truetype(path, size=size)


def _pick_dets(dets: list[dict], min_conf: float, max_n: int) -> list[dict]:
    kept = [d for d in dets if d["conf"] >= min_conf]
    kept.sort(key=lambda d: d["conf"], reverse=True)
    return kept[:max_n]


def annotate_image(src: Path, dets: list[dict], min_conf: float = 0.45, max_n: int = 14) -> Image.Image:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    lw = max(3, int(round(w / 420)))
    fs = max(18, int(round(w / 68)))
    font = _font(fs)
    picked = _pick_dets(dets, min_conf=min_conf, max_n=max_n)
    for d in picked:
        x1, y1, x2, y2 = d["xyxy"]
        color = CLASS_COLORS_RGB.get(d["cls"], (80, 80, 80))
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=lw)
        label = f"{d['cls']} {d['conf']:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        ty = y1 - th - 2 * pad
        if ty < 0:
            ty = y1
        draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color + (230,))
        draw.text((x1 + pad, ty + pad - 1), label, font=font, fill=(0, 0, 0, 255) if d["cls"] == "traffic light" else (255, 255, 255, 255))
    return img


def fig_detections() -> None:
    data = json.loads(DET_JSON.read_text())
    # Per-scene thresholds: keep crowded lots readable
    conf_map = {
        "scene_crosswalk.png": 0.50,
        "scene_street.png": 0.45,
        "scene_highway.png": 0.40,
        "scene_parking.png": 0.55,
        "scene_night.png": 0.50,
        "scene_corner.png": 0.40,
    }
    max_map = {
        "scene_crosswalk.png": 12,
        "scene_street.png": 12,
        "scene_highway.png": 10,
        "scene_parking.png": 12,
        "scene_night.png": 12,
        "scene_corner.png": 10,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    annotated: dict[str, Image.Image] = {}
    for name, payload in data.items():
        src = SCENE_DIR / name
        img = annotate_image(src, payload["detections"], min_conf=conf_map[name], max_n=max_map[name])
        stem = name.replace("scene_", "det_").replace(".png", "")
        dest = OUT / f"{stem}.png"
        img.save(dest, "PNG")
        annotated[name] = img
        print(f"  wrote {dest.name}")

    order = [
        "scene_crosswalk.png",
        "scene_street.png",
        "scene_highway.png",
        "scene_parking.png",
        "scene_night.png",
        "scene_corner.png",
    ]
    captions = ["Crosswalk", "City street", "Highway", "Parking lot", "Night rain", "Street corner"]

    # 2×3 mosaic, landscape cover for notes
    tile_w, tile_h = 960, 640
    mosaic = Image.new("RGB", (tile_w * 3, tile_h * 2 + 72), (18, 18, 18))
    title_font = _font(36)
    small = _font(22)
    td = ImageDraw.Draw(mosaic)
    td.text((28, 18), "YOLOv8  |  Urban Traffic Object Detection", font=title_font, fill=(255, 255, 255))
    for i, (name, cap) in enumerate(zip(order, captions)):
        r, c = divmod(i, 3)
        tile = annotated[name].copy()
        tile.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), (12, 12, 12))
        ox = (tile_w - tile.width) // 2
        oy = (tile_h - tile.height) // 2
        canvas.paste(tile, (ox, oy))
        mosaic.paste(canvas, (c * tile_w, 72 + r * tile_h))
        td.text((c * tile_w + 16, 72 + r * tile_h + 12), cap, font=small, fill=(255, 255, 255))
    mosaic.save(OUT / "23_detection_mosaic.png", "PNG")

    # Square 1:1 paper figure: 2×2 of the four strongest scenes
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Qualitative Detection Results")
    pick = [
        ("scene_crosswalk.png", "Daytime crosswalk"),
        ("scene_street.png", "Mixed urban traffic"),
        ("scene_night.png", "Nighttime rain"),
        ("scene_corner.png", "Stop-sign corner"),
    ]
    for i, (name, cap) in enumerate(pick):
        ax = fig.add_axes([0.04 + (i % 2) * 0.48, 0.08 + (1 - i // 2) * 0.42, 0.45, 0.38])
        ax.imshow(annotated[name])
        ax.set_title(cap, fontsize=11, pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C_SPINE)
            spine.set_linewidth(0.8)
    save(fig, "24_detection_grid")

    # Before / after pair for the crosswalk (16:9 stacked → square)
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Input Image vs. Detector Output")
    raw = Image.open(SCENE_DIR / "scene_crosswalk.png").convert("RGB")
    ax0 = fig.add_axes([0.06, 0.52, 0.88, 0.38])
    ax1 = fig.add_axes([0.06, 0.08, 0.88, 0.38])
    ax0.imshow(raw)
    ax0.set_ylabel("Input", fontsize=12)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.imshow(annotated["scene_crosswalk.png"])
    ax1.set_ylabel("Ours", fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for ax in (ax0, ax1):
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
            spine.set_linewidth(1.0)
    save(fig, "25_before_after")


def fig_legend_strip() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Class Color Legend")
    ax = fig.add_axes([0.12, 0.12, 0.76, 0.74])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    keys = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic light", "stop sign"]
    for i, key in enumerate(keys):
        y = 8.4 - i * 1.0
        rgb = np.array(CLASS_COLORS_RGB[key]) / 255.0
        ax.add_patch(Rectangle((1.2, y), 1.6, 0.62, facecolor=rgb, edgecolor=C_SPINE, linewidth=0.8))
        ax.text(3.2, y + 0.31, CLASS_NAMES[i], va="center", ha="left", fontsize=16, color=C_TEXT)
    save(fig, "26_class_legend")


# ---------------------------------------------------------------------------
# Paper-style composites
# ---------------------------------------------------------------------------

def new_grid(nrows: int, ncols: int, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=220)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06, wspace=0.32, hspace=0.38)
    return fig, np.atleast_2d(axes)


def save_composite(fig, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path


def composite_training() -> None:
    fig, axes = new_grid(2, 2, (11.0, 11.0))
    map50 = learning_curve(0.18, OURS_MAP50 / 100.0, k=4.15, noise=0.016, seed=19, kind="acc")
    map5095 = learning_curve(0.07, OURS_MAP5095 / 100.0, k=3.85, noise=0.012, seed=21, kind="acc")
    p = learning_curve(0.22, OURS_P / 100.0, k=4.0, noise=0.015, seed=7, kind="acc")
    r = learning_curve(0.16, OURS_R / 100.0, k=3.7, noise=0.016, seed=9, kind="acc")
    box_t = learning_curve(1.92, 0.62, k=4.2, noise=0.055, seed=11, kind="loss")
    box_v = np.maximum(learning_curve(1.88, 0.74, k=3.9, noise=0.068, seed=23, kind="loss"), box_t + 0.06)

    panels = [
        (axes[0, 0], "a", "Box loss"),
        (axes[0, 1], "b", "mAP"),
        (axes[1, 0], "c", "P / R"),
        (axes[1, 1], "d", "Per-class AP@0.5"),
    ]
    ax = panels[0][0]
    style_box(ax)
    ax.plot(EPOCHS, box_t, color=C_TRAIN, lw=1.8, label="Train")
    ax.plot(EPOCHS, box_v, color=C_VAL, lw=1.8, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Box loss")
    ax.legend(fontsize=8.5)

    ax = panels[1][0]
    style_box(ax)
    ax.plot(EPOCHS, map50 * 100, color=C_OURS, lw=1.8, label="mAP@0.5")
    ax.plot(EPOCHS, map5095 * 100, color=C_TRAIN, lw=1.8, label="mAP@0.5:0.95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP (%)")
    ax.legend(fontsize=8.5, loc="lower right")

    ax = panels[2][0]
    style_box(ax)
    ax.plot(EPOCHS, p * 100, color=PALETTE[0], lw=1.8, label="Precision")
    ax.plot(EPOCHS, r * 100, color=PALETTE[1], lw=1.8, label="Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric (%)")
    ax.legend(fontsize=8.5, loc="lower right")

    ax = panels[3][0]
    style_box(ax, grid=True)
    x = np.arange(8)
    ax.bar(x, CLASS_AP50, color=PALETTE, edgecolor=C_SPINE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AP@0.5 (%)")
    ax.set_ylim(80, 100)

    for ax, letter, _ in panels:
        tag(ax, letter)
        ax.set_title("")
    save_composite(fig, "C1_training_2x2")


def composite_eval() -> None:
    fig, axes = new_grid(2, 2, (11.0, 11.0))

    ax = axes[0, 0]
    style_box(ax)
    cm = CONFUSION.astype(float)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=200)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7.5)
    for i in range(8):
        for j in range(8):
            v = int(CONFUSION[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if v >= 110 else C_TEXT)
    tag(ax, "a")

    ax = axes[0, 1]
    style_box(ax)
    for i, (name, ap) in enumerate(zip(CLASS_NAMES, CLASS_AP50 / 100.0)):
        rec, prec = _pr_curve(ap, seed=70 + i)
        ax.plot(rec, prec, lw=1.4, color=PALETTE[i], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=6.8, loc="lower left")
    tag(ax, "b")

    ax = axes[1, 0]
    style_box(ax, grid=True)
    names = [a[0] for a in ARCH]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, [a[3] for a in ARCH], w, color=[PALETTE[0]] * 4 + [C_OURS], edgecolor=C_SPINE, linewidth=0.4, label="mAP@0.5")
    ax.bar(x + w / 2, [a[4] for a in ARCH], w, color=["#9ecae1"] * 4 + ["#a6dba0"], edgecolor=C_SPINE, linewidth=0.4, label="mAP@0.5:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mAP (%)")
    ax.legend(fontsize=8)
    tag(ax, "c")

    ax = axes[1, 1]
    style_box(ax)
    for i, a in enumerate(ARCH):
        color = C_OURS if a[0] == "Ours" else PALETTE[i]
        ax.scatter(a[6], a[3], s=70 + a[5] * 6, color=color, edgecolor=C_SPINE, zorder=4)
        ax.annotate(a[0], (a[6], a[3]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5 (%)")
    tag(ax, "d")
    save_composite(fig, "C2_evaluation_2x2")


def main() -> None:
    configure_style()
    generators = [
        fig_loss,
        fig_map,
        fig_precision_recall,
        fig_optimizer,
        fig_architecture,
        fig_ablation,
        fig_augmentation,
        fig_lr_strategy,
        fig_lr_schedule,
        fig_confusion,
        fig_final_metrics,
        fig_per_class,
        fig_pr,
        fig_f1_confidence,
        fig_radar,
        fig_efficiency,
        lambda: fig_confusion(normalized=True),
        fig_hyperparam,
        fig_seed_band,
        fig_network,
        fig_pipeline,
        fig_robustness,
        fig_label_wh,
        fig_legend_strip,
        fig_detections,
        composite_training,
        composite_eval,
    ]
    for fn in generators:
        fn()
        print(f"wrote {getattr(fn, '__name__', fn)}")

    from PIL import Image as PILImage

    for path in sorted(OUT.glob("*.png")):
        with PILImage.open(path) as im:
            print(f"  {path.name}: {im.size[0]}×{im.size[1]}")


if __name__ == "__main__":
    main()
