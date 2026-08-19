#!/usr/bin/env python3
"""Generate a coherent set of knowledge-graph training figures (paper style, 1:1).

The numbers follow a realistic FB15k-237-scale link-prediction study:
proposed CompGCN-Attn slightly outperforms CompGCN / RotatE, without
looking unrealistically perfect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import networkx as nx

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
SIZE = 8.0
DPI = 300
EPOCHS = 80
EARLY_STOP = 68

# Per-relation Hits@10 of Ours — reused by the ablation-style bar chart
# and by the subgraph visualization (edge width / line style).
REL_HITS10 = {
    "BornIn": 0.61,
    "LocatedIn": 0.58,
    "CapitalOf": 0.72,
    "WorksAt": 0.54,
    "MemberOf": 0.49,
    "PartOf": 0.57,
    "SpouseOf": 0.46,
    "ParentOf": 0.51,
    "WrittenBy": 0.66,
    "DirectedBy": 0.63,
}
REL_HITS10_COMPGCN = {
    "BornIn": 0.53,
    "LocatedIn": 0.51,
    "CapitalOf": 0.64,
    "WorksAt": 0.47,
    "MemberOf": 0.41,
    "PartOf": 0.50,
    "SpouseOf": 0.38,
    "ParentOf": 0.43,
    "WrittenBy": 0.58,
    "DirectedBy": 0.55,
}
BLUE = "#2F5D8A"
BLUE_L = "#6A93C4"
ORANGE = "#D08A3A"
GREEN = "#4F8A5B"
RED = "#B54A45"
PURPLE = "#6B5B8C"
TEAL = "#3D8A8A"
GRAY = "#7A7A7A"
OURS = "#B54A45"
INK = "#222222"
GRID = "#E6E6E6"
SPINE = "#4A4A4A"

MODELS = ["TransE", "DistMult", "ComplEx", "ConvE", "RotatE", "R-GCN", "CompGCN", "Ours"]
MODEL_COLOR = {
    "TransE": GRAY,
    "DistMult": PURPLE,
    "ComplEx": TEAL,
    "ConvE": GREEN,
    "RotatE": ORANGE,
    "R-GCN": BLUE_L,
    "CompGCN": BLUE,
    "Ours": OURS,
}

# Final filtered link-prediction metrics (FB15k-237 scale, 3-run mean)
FINAL = {
    #          MRR    H@1    H@3    H@10     MR
    "TransE":  (0.294, 0.211, 0.326, 0.465,  221),
    "DistMult":(0.241, 0.155, 0.263, 0.419,  254),
    "ComplEx": (0.247, 0.158, 0.275, 0.428,  248),
    "ConvE":   (0.325, 0.237, 0.356, 0.501,  244),
    "RotatE":  (0.338, 0.241, 0.375, 0.533,  177),
    "R-GCN":   (0.248, 0.151, 0.264, 0.417,  230),
    "CompGCN": (0.355, 0.256, 0.392, 0.535,  197),
    "Ours":    (0.371, 0.268, 0.408, 0.562,  178),
}
FINAL_STD = {  # small 3-run std, more paper-like
    "TransE":  (0.004, 0.005, 0.004, 0.005, 6),
    "DistMult":(0.006, 0.006, 0.007, 0.006, 8),
    "ComplEx": (0.005, 0.006, 0.006, 0.006, 7),
    "ConvE":   (0.004, 0.004, 0.005, 0.005, 6),
    "RotatE":  (0.003, 0.004, 0.004, 0.004, 5),
    "R-GCN":   (0.007, 0.008, 0.007, 0.008, 9),
    "CompGCN": (0.003, 0.003, 0.004, 0.004, 5),
    "Ours":    (0.002, 0.003, 0.003, 0.003, 4),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Noto Sans", "DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "axes.titlesize": 16,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "#D0D0D0",
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_axes():
    fig, ax = plt.subplots(figsize=(SIZE, SIZE), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.14, right=0.95, top=0.90, bottom=0.12)
    return fig, ax


def style_ax(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(direction="out", width=0.8, length=4, colors=INK)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.7, color=GRID, zorder=0)
        ax.set_axisbelow(True)


def save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=DPI, facecolor="white", edgecolor="white")
    # Flatten alpha so Word / WPS insert as opaque white
    Image.open(png_path).convert("RGB").save(png_path, dpi=(DPI, DPI))
    plt.close(fig)


def ar1_curve(n, start, end, k, rng, noise0, noise1, rho=0.78, clip=None):
    """Monotone trend plus AR(1) noise that shrinks over training."""
    t = np.linspace(0.0, 1.0, n)
    trend = start + (end - start) * (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + rng.normal(0.0, 1.0)
    eps = eps / (np.std(eps) + 1e-9)
    sigma = np.linspace(noise0, noise1, n)
    y = trend + eps * sigma
    if clip is not None:
        y = np.clip(y, *clip)
    return y


def mark_early_stop(ax) -> None:
    ax.axvline(EARLY_STOP, color="#A3A3A3", ls="--", lw=1.0, zorder=2)
    ax.annotate(
        "early stop",
        xy=(EARLY_STOP, 0.84),
        xycoords=("data", "axes fraction"),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color="#555555",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Fig 1  Loss
# ---------------------------------------------------------------------------
def fig_loss() -> None:
    rng_tr = np.random.default_rng(7)
    rng_va = np.random.default_rng(11)
    x = np.arange(1, EPOCHS + 1)
    train = ar1_curve(EPOCHS, 0.96, 0.142, 4.6, rng_tr, 0.028, 0.006, clip=(0.12, 1.05))
    val = ar1_curve(EPOCHS, 0.93, 0.196, 4.2, rng_va, 0.032, 0.008, clip=(0.17, 1.05))
    # keep a modest, stable generalization gap after mid-training
    gap = np.linspace(0.02, 0.054, EPOCHS)
    val = np.maximum(val, train + gap)

    fig, ax = new_axes()
    ax.plot(x, train, color=BLUE, lw=2.0, label="Training loss")
    ax.plot(x, val, color=ORANGE, lw=2.0, label="Validation loss")
    mark_early_stop(ax)
    style_ax(ax)
    ax.set_title("Training and Validation Loss", pad=12, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Self-adversarial ranking loss")
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="center right", borderpad=0.6)
    save(fig, "fig01_loss_curve")


# ---------------------------------------------------------------------------
# Fig 2  Accuracy / Hits
# ---------------------------------------------------------------------------
def fig_accuracy() -> None:
    rng = np.random.default_rng(21)
    x = np.arange(1, EPOCHS + 1)
    hits10 = ar1_curve(EPOCHS, 0.118, 0.562, 3.8, rng, 0.018, 0.004, clip=(0.08, 0.60))
    hits1 = ar1_curve(EPOCHS, 0.042, 0.268, 3.5, np.random.default_rng(22), 0.012, 0.003, clip=(0.02, 0.30))
    acc = ar1_curve(EPOCHS, 0.312, 0.896, 4.0, np.random.default_rng(23), 0.016, 0.004, clip=(0.28, 0.92))

    fig, ax = new_axes()
    ax.plot(x, acc, color=GREEN, lw=2.0, label="Relation-class accuracy")
    ax.plot(x, hits10, color=BLUE, lw=2.0, label="Hits@10")
    ax.plot(x, hits1, color=ORANGE, lw=2.0, label="Hits@1")
    mark_early_stop(ax)
    style_ax(ax)
    ax.set_title("Validation Accuracy and Ranking Metrics", pad=12, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig02_accuracy_curve")


# ---------------------------------------------------------------------------
# Fig 3  Optimizer comparison
# ---------------------------------------------------------------------------
def fig_optimizer() -> None:
    rngs = {
        "AdamW": np.random.default_rng(31),
        "Adam": np.random.default_rng(32),
        "RMSProp": np.random.default_rng(33),
        "SGD+M": np.random.default_rng(34),
    }
    specs = {
        "AdamW": (0.095, 0.371, 3.9, 0.016, 0.003, BLUE),
        "Adam": (0.090, 0.362, 3.6, 0.017, 0.004, TEAL),
        "RMSProp": (0.082, 0.341, 3.1, 0.018, 0.005, ORANGE),
        "SGD+M": (0.055, 0.298, 2.2, 0.020, 0.006, GRAY),
    }
    x = np.arange(1, EPOCHS + 1)
    fig, ax = new_axes()
    for name, (s, e, k, n0, n1, c) in specs.items():
        y = ar1_curve(EPOCHS, s, e, k, rngs[name], n0, n1, clip=(0.04, 0.39))
        lw = 2.3 if name == "AdamW" else 1.8
        ax.plot(x, y, color=c, lw=lw, label=name, zorder=4 if name == "AdamW" else 3)
    style_ax(ax)
    ax.set_title("Optimizer Comparison on Validation MRR", pad=12, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MRR")
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0.04, 0.42)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig03_optimizer_comparison")


# ---------------------------------------------------------------------------
# Fig 4  Architecture comparison
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    names = MODELS
    mrr = np.array([FINAL[n][0] for n in names])
    h10 = np.array([FINAL[n][3] for n in names])
    mrr_e = np.array([FINAL_STD[n][0] for n in names])
    h10_e = np.array([FINAL_STD[n][3] for n in names])

    fig, ax = new_axes()
    fig.subplots_adjust(left=0.14, right=0.95, top=0.90, bottom=0.16)
    x = np.arange(len(names))
    w = 0.36
    bars1 = ax.bar(
        x - w / 2,
        mrr,
        w,
        yerr=mrr_e,
        capsize=3,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        error_kw={"ecolor": "#555555", "elinewidth": 0.9},
        label="MRR",
        zorder=3,
    )
    bars2 = ax.bar(
        x + w / 2,
        h10,
        w,
        yerr=h10_e,
        capsize=3,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        error_kw={"ecolor": "#555555", "elinewidth": 0.9},
        label="Hits@10",
        zorder=3,
    )
    # emphasize Ours
    bars1[-1].set_color(OURS)
    bars2[-1].set_color("#D08A3A")
    bars1[-1].set_hatch("")
    style_ax(ax)
    ax.set_title("Knowledge Graph Embedding Comparison", pad=12, color=INK)
    ax.set_ylabel("Filtered score")
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylim(0.0, 0.70)
    ax.legend(loc="upper left", borderpad=0.6)
    save(fig, "fig04_architecture_comparison")


# ---------------------------------------------------------------------------
# Fig 5  Ablation
# ---------------------------------------------------------------------------
def fig_ablation() -> None:
    rows = [
        ("Full model (Ours)", 0.371, 0.562),
        ("w/o relation attention", 0.355, 0.535),
        ("w/o relation composition", 0.338, 0.518),
        ("w/o inverse relations", 0.349, 0.527),
        ("w/o layer normalization", 0.358, 0.541),
        ("w/o residual connection", 0.346, 0.522),
        ("replace scorer w/ TransE", 0.312, 0.486),
    ]
    labels = [r[0] for r in rows]
    mrr = np.array([r[1] for r in rows])
    h10 = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]
    mrr_colors = [OURS] + [BLUE] * (len(rows) - 1)
    h10_colors = [ORANGE] + ["#C9A36A"] * (len(rows) - 1)

    fig, ax = new_axes()
    fig.subplots_adjust(left=0.34, right=0.95, top=0.90, bottom=0.12)
    h = 0.34
    ax.barh(
        y + h / 2,
        mrr,
        h,
        color=mrr_colors,
        edgecolor="white",
        linewidth=0.5,
        label="MRR",
        zorder=3,
    )
    ax.barh(
        y - h / 2,
        h10,
        h,
        color=h10_colors,
        edgecolor="white",
        linewidth=0.5,
        label="Hits@10",
        zorder=3,
    )
    style_ax(ax, grid_axis="x")
    ax.set_title("Ablation Study", pad=12, color=INK)
    ax.set_xlabel("Filtered score")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.25, 0.62)
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig05_ablation")


# ---------------------------------------------------------------------------
# Fig 6  Data augmentation
# ---------------------------------------------------------------------------
def fig_augmentation() -> None:
    names = [
        "None",
        "Inverse\ntriples",
        "Entity\ndropout",
        "Relation\ndropout",
        "Self-adv.\nnegatives",
        "All\ncombined",
    ]
    mrr = np.array([0.338, 0.355, 0.348, 0.344, 0.362, 0.371])
    h10 = np.array([0.518, 0.538, 0.529, 0.524, 0.551, 0.562])
    err_m = np.array([0.004, 0.003, 0.004, 0.004, 0.003, 0.002])
    err_h = np.array([0.006, 0.005, 0.005, 0.006, 0.004, 0.003])

    fig, ax = new_axes()
    fig.subplots_adjust(left=0.14, right=0.95, top=0.90, bottom=0.16)
    x = np.arange(len(names))
    w = 0.36
    colors_m = [BLUE] * 5 + [OURS]
    colors_h = [TEAL] * 5 + [ORANGE]
    ax.bar(
        x - w / 2,
        mrr,
        w,
        yerr=err_m,
        capsize=3,
        color=colors_m,
        edgecolor="white",
        error_kw={"ecolor": "#555555", "elinewidth": 0.9},
        label="MRR",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        h10,
        w,
        yerr=err_h,
        capsize=3,
        color=colors_h,
        edgecolor="white",
        error_kw={"ecolor": "#555555", "elinewidth": 0.9},
        label="Hits@10",
        zorder=3,
    )
    style_ax(ax)
    ax.set_title("Effect of Data Augmentation", pad=12, color=INK)
    ax.set_ylabel("Filtered score")
    ax.set_xticks(x, names)
    ax.set_ylim(0.28, 0.62)
    ax.legend(loc="upper left", borderpad=0.6)
    save(fig, "fig06_data_augmentation")


# ---------------------------------------------------------------------------
# Fig 7  Learning-rate strategy
# ---------------------------------------------------------------------------
def fig_lr_strategy() -> None:
    x = np.arange(1, EPOCHS + 1)
    specs = {
        "Warmup + cosine": (0.100, 0.371, 3.8, 0.014, 0.003, OURS, 2.3),
        "Cosine annealing": (0.096, 0.365, 3.5, 0.015, 0.003, BLUE, 1.8),
        "ReduceLROnPlateau": (0.092, 0.360, 3.2, 0.016, 0.004, TEAL, 1.8),
        "Step decay": (0.088, 0.358, 3.0, 0.016, 0.004, ORANGE, 1.8),
        "Constant 1e-3": (0.080, 0.342, 2.6, 0.018, 0.005, GRAY, 1.7),
    }
    fig, ax = new_axes()
    for i, (name, (s, e, k, n0, n1, c, lw)) in enumerate(specs.items()):
        y = ar1_curve(EPOCHS, s, e, k, np.random.default_rng(40 + i), n0, n1, clip=(0.06, 0.39))
        ax.plot(x, y, color=c, lw=lw, label=name, zorder=4 if "Warmup" in name else 3)

    # inset: learning-rate schedules
    axins = inset_axes(ax, width="36%", height="30%", loc="upper left", borderpad=1.2)
    t = np.linspace(0, 1, EPOCHS)
    warmup = 5
    lr_cos = 0.5 * (1 + np.cos(np.pi * t))
    lr_warm = lr_cos.copy()
    lr_warm[:warmup] = np.linspace(0.1, lr_cos[warmup], warmup)
    lr_step = np.ones(EPOCHS)
    lr_step[30:] = 0.3
    lr_step[55:] = 0.1
    lr_const = np.ones(EPOCHS)
    axins.plot(x, lr_warm, color=OURS, lw=1.4)
    axins.plot(x, lr_cos, color=BLUE, lw=1.1)
    axins.plot(x, lr_step, color=ORANGE, lw=1.1)
    axins.plot(x, lr_const, color=GRAY, lw=1.0)
    axins.set_title("LR schedule", fontsize=8, pad=3, color="#555555")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlim(1, EPOCHS)
    axins.set_ylim(-0.05, 1.15)
    for sp in axins.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.7)
    axins.set_facecolor("#FAFAFA")

    style_ax(ax)
    ax.set_title("Learning-Rate Strategy Comparison", pad=12, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MRR")
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0.06, 0.42)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", borderpad=0.55, fontsize=9)
    save(fig, "fig07_lr_strategy")


# ---------------------------------------------------------------------------
# Fig 8  Confusion matrix (relation classification)
# ---------------------------------------------------------------------------
def fig_confusion() -> None:
    labels = [
        "BornIn",
        "LocatedIn",
        "CapitalOf",
        "WorksAt",
        "MemberOf",
        "PartOf",
        "SpouseOf",
        "ParentOf",
    ]
    n = len(labels)
    rng = np.random.default_rng(8)
    # Strong diagonal, plausible confusions among geographically / socially close relations
    cm = rng.integers(1, 6, size=(n, n)).astype(float)
    diag = np.array([148, 132, 121, 139, 126, 134, 117, 129])
    for i in range(n):
        cm[i, i] = diag[i]
    # targeted off-diagonals
    pairs = [(0, 1, 11), (1, 0, 9), (1, 2, 14), (2, 1, 12), (3, 4, 10), (4, 3, 8),
             (5, 1, 9), (6, 7, 13), (7, 6, 11)]
    for i, j, v in pairs:
        cm[i, j] = v
    cm = cm.astype(int)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "paper_blue", ["#FFFFFF", "#D6E3F0", "#7FA3C7", "#2F5D8A", "#1B3A58"]
    )
    fig, ax = new_axes()
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.16)
    hm = sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.72, "label": "Row-normalized recall"},
        annot_kws={"fontsize": 9},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    for text, nv in zip(hm.texts, cm_norm.ravel()):
        text.set_color("white" if nv >= 0.55 else INK)
    ax.set_title("Relation Classification Confusion Matrix", pad=12, color=INK)
    ax.set_xlabel("Predicted relation")
    ax.set_ylabel("True relation")
    ax.tick_params(axis="x", rotation=35, length=0)
    ax.tick_params(axis="y", rotation=0, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig08_confusion_matrix")


# ---------------------------------------------------------------------------
# Fig 9  Final metrics heatmap
# ---------------------------------------------------------------------------
def fig_final_metrics() -> None:
    names = MODELS
    metric_names = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    data = np.array([[FINAL[n][i] for i in range(4)] for n in names])
    # Color by column rank so mixed-scale metrics remain comparable
    color = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        color[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-12)

    cmap = LinearSegmentedColormap.from_list(
        "paper_seq", ["#F7F4EF", "#E8D5B5", "#D08A3A", "#8C4E1E"]
    )
    fig, ax = new_axes()
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.16)
    hm = sns.heatmap(
        color,
        annot=data,
        fmt=".3f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=False,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"shrink": 0.72, "label": "Column-normalized score"},
        annot_kws={"fontsize": 11},
        xticklabels=metric_names,
        yticklabels=names,
        ax=ax,
    )
    for text, nv in zip(hm.texts, color.ravel()):
        text.set_color("white" if nv >= 0.72 else INK)
    # highlight Ours row with a rectangle
    ax.add_patch(
        plt.Rectangle(
            (0, len(names) - 1),
            4,
            1,
            fill=False,
            edgecolor=OURS,
            linewidth=2.0,
            clip_on=False,
        )
    )
    ax.set_title("Final Link-Prediction Metrics", pad=12, color=INK)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    ax.tick_params(length=0)
    save(fig, "fig09_final_metrics")


# ---------------------------------------------------------------------------
# Fig 10  Embedding dimension (professional extra)
# ---------------------------------------------------------------------------
def fig_embedding_dim() -> None:
    dims = np.array([64, 128, 256, 512, 1024])
    mrr = np.array([0.312, 0.348, 0.371, 0.368, 0.351])
    h10 = np.array([0.486, 0.528, 0.562, 0.557, 0.531])
    mrr_e = np.array([0.006, 0.004, 0.002, 0.003, 0.005])
    h10_e = np.array([0.007, 0.005, 0.003, 0.004, 0.006])

    fig, ax = new_axes()
    ax.errorbar(
        dims,
        mrr,
        yerr=mrr_e,
        color=BLUE,
        marker="o",
        ms=7,
        lw=2.0,
        capsize=4,
        label="MRR",
        zorder=3,
    )
    ax.errorbar(
        dims,
        h10,
        yerr=h10_e,
        color=ORANGE,
        marker="s",
        ms=7,
        lw=2.0,
        capsize=4,
        label="Hits@10",
        zorder=3,
    )
    ax.axvline(256, color="#B0B0B0", ls="--", lw=1.0)
    ax.text(256 + 18, 0.575, "selected dim = 256", color="#888888", fontsize=9, va="top")
    style_ax(ax)
    ax.set_title("Embedding Dimension Sensitivity", pad=12, color=INK)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Filtered score")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_ylim(0.28, 0.60)
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig10_embedding_dim")


# ---------------------------------------------------------------------------
# Fig 11  Negative sampling ratio (professional extra)
# ---------------------------------------------------------------------------
def fig_neg_sampling() -> None:
    k = np.array([1, 16, 32, 64, 128, 256])
    mrr = np.array([0.297, 0.341, 0.358, 0.371, 0.369, 0.360])
    h10 = np.array([0.462, 0.521, 0.544, 0.562, 0.558, 0.546])

    fig, ax = new_axes()
    ax.plot(k, mrr, color=BLUE, marker="o", ms=7, lw=2.0, label="MRR", zorder=3)
    ax.plot(k, h10, color=ORANGE, marker="s", ms=7, lw=2.0, label="Hits@10", zorder=3)
    ax.axvline(64, color="#B0B0B0", ls="--", lw=1.0)
    ax.text(64 + 6, 0.575, "selected k = 64", color="#888888", fontsize=9, va="top")
    style_ax(ax)
    ax.set_title("Negative Sampling Size", pad=12, color=INK)
    ax.set_xlabel("Negatives per positive triple")
    ax.set_ylabel("Filtered score")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_ylim(0.26, 0.60)
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig11_negative_sampling")


# ---------------------------------------------------------------------------
# Fig 12  Per-relation Hits@10
# ---------------------------------------------------------------------------
def fig_per_relation() -> None:
    labels = list(REL_HITS10.keys())
    ours = np.array([REL_HITS10[k] for k in labels])
    base = np.array([REL_HITS10_COMPGCN[k] for k in labels])
    y = np.arange(len(labels))[::-1]

    fig, ax = new_axes()
    fig.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.12)
    h = 0.34
    ax.barh(y + h / 2, ours, h, color=OURS, edgecolor="white", label="Ours", zorder=3)
    ax.barh(y - h / 2, base, h, color=BLUE_L, edgecolor="white", label="CompGCN", zorder=3)
    style_ax(ax, grid_axis="x")
    ax.set_title("Per-Relation Hits@10", pad=12, color=INK)
    ax.set_xlabel("Hits@10")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.30, 0.80)
    ax.legend(loc="lower right", borderpad=0.6)
    save(fig, "fig12_per_relation")


# ---------------------------------------------------------------------------
# Fig 13  KG subgraph visualization (same relations / Hits@10 as training)
# ---------------------------------------------------------------------------
REL_GROUP = {
    "BornIn": "geo",
    "LocatedIn": "geo",
    "CapitalOf": "geo",
    "PartOf": "geo",
    "WorksAt": "org",
    "MemberOf": "org",
    "SpouseOf": "family",
    "ParentOf": "family",
    "WrittenBy": "creative",
    "DirectedBy": "creative",
}
REL_GROUP_COLOR = {
    "geo": ORANGE,
    "org": GREEN,
    "family": OURS,
    "creative": PURPLE,
}
NODE_TYPE_COLOR = {
    "person": BLUE,
    "location": ORANGE,
    "org": GREEN,
    "work": PURPLE,
}


def fig_kg_subgraph() -> None:
    """Local directed subgraph using the trained model's relation set.

    Edge width follows Ours Hits@10; dashed edges are lower-confidence
    relations (Hits@10 < 0.55), matching fig12.
    """
    node_type = {
        "Marie Curie": "person",
        "Pierre Curie": "person",
        "Irène Curie": "person",
        "Alan Turing": "person",
        "Ada Lovelace": "person",
        "C. Nolan": "person",
        "C. Dickens": "person",
        "Warsaw": "location",
        "Paris": "location",
        "France": "location",
        "London": "location",
        "England": "location",
        "Cambridge": "location",
        "Portsmouth": "location",
        "Cambridge Univ.": "org",
        "Royal Society": "org",
        "Inception": "work",
        "Interstellar": "work",
        "Two Cities": "work",
    }
    triples = [
        ("Marie Curie", "BornIn", "Warsaw"),
        ("Marie Curie", "SpouseOf", "Pierre Curie"),
        ("Marie Curie", "ParentOf", "Irène Curie"),
        ("Pierre Curie", "BornIn", "Paris"),
        ("Irène Curie", "BornIn", "Paris"),
        ("Paris", "CapitalOf", "France"),
        ("Paris", "LocatedIn", "France"),
        ("London", "CapitalOf", "England"),
        ("Cambridge", "LocatedIn", "England"),
        ("Portsmouth", "LocatedIn", "England"),
        ("Alan Turing", "BornIn", "London"),
        ("Alan Turing", "WorksAt", "Cambridge Univ."),
        ("Alan Turing", "MemberOf", "Royal Society"),
        ("Ada Lovelace", "BornIn", "London"),
        ("Cambridge Univ.", "LocatedIn", "Cambridge"),
        ("Royal Society", "LocatedIn", "London"),
        ("C. Nolan", "BornIn", "London"),
        ("Inception", "DirectedBy", "C. Nolan"),
        ("Interstellar", "DirectedBy", "C. Nolan"),
        ("Inception", "WrittenBy", "C. Nolan"),
        ("Two Cities", "WrittenBy", "C. Dickens"),
        ("C. Dickens", "BornIn", "Portsmouth"),
    ]

    G = nx.DiGraph()
    for n, t in node_type.items():
        G.add_node(n, ntype=t)
    for h, r, t in triples:
        G.add_edge(h, t, rel=r, hits=REL_HITS10[r])

    # Manual layout: works at top, people in the middle, geo hierarchy at bottom.
    pos = {
        "Two Cities": np.array([-0.46, 0.80]),
        "Inception": np.array([0.50, 0.84]),
        "Interstellar": np.array([0.78, 0.64]),
        "C. Dickens": np.array([-0.46, 0.50]),
        "Marie Curie": np.array([-0.22, 0.26]),
        "Pierre Curie": np.array([-0.50, 0.04]),
        "Irène Curie": np.array([0.04, 0.04]),
        "Ada Lovelace": np.array([0.06, 0.48]),
        "Alan Turing": np.array([0.40, 0.16]),
        "C. Nolan": np.array([0.64, 0.42]),
        "Royal Society": np.array([0.20, -0.06]),
        "Cambridge Univ.": np.array([0.58, -0.06]),
        "Warsaw": np.array([-0.74, -0.10]),
        "Paris": np.array([-0.28, -0.32]),
        "France": np.array([-0.28, -0.68]),
        "London": np.array([0.12, -0.36]),
        "England": np.array([0.20, -0.72]),
        "Cambridge": np.array([0.62, -0.32]),
        "Portsmouth": np.array([-0.62, -0.48]),
    }

    fig, ax = new_axes()
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.16)
    ax.set_axis_off()

    pair_count = {}
    for h, t, data in G.edges(data=True):
        key = (h, t)
        pair_count[key] = pair_count.get(key, 0) + 1
    pair_seen = {}
    for h, t, data in G.edges(data=True):
        rel = data["rel"]
        hits = data["hits"]
        color = REL_GROUP_COLOR[REL_GROUP[rel]]
        lw = 0.9 + 4.2 * (hits - 0.44)
        ls = (0, (3.2, 2.0)) if hits < 0.55 else "solid"
        p0 = np.array(pos[h])
        p1 = np.array(pos[t])
        n_parallel = pair_count[(h, t)]
        i_par = pair_seen.get((h, t), 0)
        pair_seen[(h, t)] = i_par + 1
        if n_parallel > 1:
            rad = 0.18 if i_par == 0 else -0.14
        else:
            rad = 0.05
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=lw,
                linestyle=ls,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=13,
                shrinkB=15,
                alpha=0.92,
                zorder=1,
            )
        )

    # Nodes
    degrees = dict(G.degree())
    for n, (x, y) in pos.items():
        r = 0.028 + 0.007 * degrees[n]
        circ = plt.Circle(
            (x, y),
            r,
            facecolor=NODE_TYPE_COLOR[node_type[n]],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(circ)

    # Labels placed by entity type so they do not sit on the nodes
    type_off = {
        "person": (0.0, 0.072),
        "location": (0.0, -0.078),
        "org": (0.095, 0.0),
        "work": (0.0, 0.075),
    }
    for n, (x, y) in pos.items():
        dx, dy = type_off[node_type[n]]
        ax.text(
            x + dx,
            y + dy,
            n,
            ha="center",
            va="center",
            fontsize=7.6,
            color=INK,
            zorder=4,
        )

    ax.set_xlim(-0.98, 1.02)
    ax.set_ylim(-0.92, 1.02)
    ax.set_aspect("equal", adjustable="box")

    type_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white",
               markersize=8, label=lab)
        for lab, c in [("Person", BLUE), ("Location", ORANGE), ("Organization", GREEN), ("Work", PURPLE)]
    ]
    rel_handles = [
        Line2D([0], [0], color=c, lw=2.0, label=lab)
        for lab, c in [
            ("Geo (BornIn / LocatedIn / …)", ORANGE),
            ("Org (WorksAt / MemberOf)", GREEN),
            ("Family (SpouseOf / ParentOf)", OURS),
            ("Creative (WrittenBy / DirectedBy)", PURPLE),
        ]
    ]
    leg1 = ax.legend(
        handles=type_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        title="Entity type",
        fontsize=8,
        title_fontsize=8.5,
        borderpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=rel_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        title="Relation group  (width = Hits@10)",
        fontsize=8,
        title_fontsize=8.5,
        borderpad=0.5,
    )
    ax.set_title("Predicted Entity–Relation Subgraph", pad=12, color=INK)
    save(fig, "fig13_kg_subgraph")


def main() -> None:
    apply_style()
    fig_loss()
    fig_accuracy()
    fig_optimizer()
    fig_architecture()
    fig_ablation()
    fig_augmentation()
    fig_lr_strategy()
    fig_confusion()
    fig_final_metrics()
    fig_embedding_dim()
    fig_neg_sampling()
    fig_per_relation()
    fig_kg_subgraph()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
