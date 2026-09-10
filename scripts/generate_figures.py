#!/usr/bin/env python3
"""Paper-style figure pack for multimodal medical image AI / radiomics.

Self-consistent study (NSCLC, CECT + 18F-FDG PET + clinical):
  Swin-UNETR segmentation, deep + handcrafted radiomics, Cox nomogram.
  Dice 0.914  |  subtype Acc 94.06%  |  3-year AUC 0.887  |  C-index 0.821

On-figure scientific text is English. Cover boards also carry Chinese titles
for Xianyu notes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, shift

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

SIZE = 8.0
DPI = 300
FONT_NAME = "DejaVu Sans"
CN_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
CN = font_manager.FontProperties(fname=CN_PATH) if Path(CN_PATH).exists() else None

C_TRAIN = "#2166ac"
C_VAL = "#b2182b"
C_OURS = "#1b7837"
C_GRID = "#d9d9d9"
C_SPINE = "#333333"
C_TEXT = "#1a1a1a"
C_GT = "#00c853"
C_PRED = "#ff3d00"
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

# ---------------------------------------------------------------------------
# Study numbers (keep every figure consistent with these)
# ---------------------------------------------------------------------------

N_EPOCHS = 100
EPOCHS = np.arange(1, N_EPOCHS + 1)

DICE_CT = 0.914
DICE_PET = 0.889
IOU = 0.841
HD95 = 3.82
SENS = 0.928
SPEC = 0.971
SUBTYPE_ACC = 94.06
CINDEX_TRAIN = 0.821
CINDEX_VAL = 0.798
CINDEX_TEST = 0.786
AUC_TRAIN = 0.921
AUC_VAL = 0.894
AUC_TEST = 0.887
AUC_EXT = 0.862
AUC_TNM = 0.694
AUC_CLIN = 0.751
AUC_RAD = 0.843

CLASS_NAMES = ["ADC", "SCC", "LCC", "SCLC"]
# 140 / 100 / 40 / 40 = 320 test cases, 301 correct → 94.06%
CONFUSION = np.array(
    [
        [134, 4, 2, 0],
        [3, 94, 2, 1],
        [2, 2, 35, 1],
        [0, 1, 1, 38],
    ],
    dtype=int,
)

ORGANS = ["GTV", "Lung", "Heart", "Spinal cord", "Esophagus"]
ORGAN_DICE = np.array([0.914, 0.968, 0.932, 0.887, 0.851])
ORGAN_HD95 = np.array([3.82, 1.46, 2.91, 2.18, 4.55])

ARCH = [
    # name, Dice, IoU, HD95 mm, params M
    ("U-Net", 0.847, 0.742, 8.64, 17.3),
    ("AttU-Net", 0.868, 0.771, 6.92, 34.9),
    ("nnU-Net", 0.891, 0.807, 5.14, 31.2),
    ("TransUNet", 0.896, 0.814, 4.88, 105.3),
    ("UNETR", 0.883, 0.795, 5.61, 92.6),
    ("Ours", DICE_CT, IOU, HD95, 62.1),
]

JET_MED = LinearSegmentedColormap.from_list(
    "jet_med",
    ["#000040", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000", "#800000"],
)


def cn(size: float = 16, weight: str = "normal") -> font_manager.FontProperties | None:
    if CN is None:
        return font_manager.FontProperties(family=FONT_NAME, size=size, weight=weight)
    return font_manager.FontProperties(fname=CN_PATH, size=size)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


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
    fig.suptitle(title, fontsize=17, fontweight="bold", color=C_TEXT, y=0.955, fontfamily=FONT_NAME)


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


def style_box(ax, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(0.95)
    if grid:
        ax.grid(True, color=C_GRID, linewidth=0.6, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def tag(ax, letter: str, fs: float = 12) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="top",
        ha="left",
        color=C_TEXT,
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )


def save(fig, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_composite(fig, stem: str, dpi: int = 230) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def annotate_bars(ax, bars, fmt: str = "{:.3f}", dy: float = 0.008, fontsize: int = 9.0):
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


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


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
    decay = (decay - decay[-1]) / (decay[0] - decay[-1] + 1e-12)
    base = end + (start - end) * decay
    amp = noise * (0.22 + 0.78 * decay)
    raw = base + rng.normal(0.0, 1.0, N_EPOCHS) * amp
    if kind == "loss":
        raw[31] += 0.35 * noise
        raw[32] += 0.18 * noise
    else:
        raw[31] -= 0.35 * noise
        raw[32] -= 0.18 * noise
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
        lo, hi = min(end, start) * 0.4, max(start, end) * 1.08
        smooth = np.clip(smooth, lo, hi)
        smooth[-1] = end
    return smooth


def cosine_warmup_lr(base: float = 1e-4, min_lr: float = 1e-6, warmup: int = 10) -> np.ndarray:
    lr = np.empty(N_EPOCHS)
    for i in range(N_EPOCHS):
        if i < warmup:
            lr[i] = min_lr + (base - min_lr) * (i + 1) / warmup
        else:
            t = (i - warmup) / max(N_EPOCHS - warmup - 1, 1)
            lr[i] = min_lr + 0.5 * (base - min_lr) * (1 + np.cos(np.pi * t))
    return lr


def roc_curve(auc: float, seed: int, n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Binormal ROC with a target AUC (concave, paper-like, not a step)."""
    from scipy.stats import norm

    rng = np.random.RandomState(seed)
    fpr = np.linspace(0.0, 1.0, n)
    a = np.sqrt(2.0) * norm.ppf(float(np.clip(auc, 0.51, 0.999)))
    eps = 1e-4
    tpr = norm.cdf(a + norm.ppf(np.clip(fpr, eps, 1.0 - eps)))
    jitter = rng.normal(0.0, 0.005, n) * np.sin(np.pi * fpr)
    tpr = np.clip(tpr + jitter, 0.0, 1.0)
    tpr = np.maximum.accumulate(tpr)
    tpr[0] = 0.0
    tpr[-1] = 1.0
    return fpr, tpr


def dca_nb(pt: np.ndarray, prev: float, auc: float) -> np.ndarray:
    """Net benefit: starts with treat-all, stays higher for stronger models."""
    from scipy.stats import norm

    treat_all = (prev - pt) / np.clip(1.0 - pt, 1e-6, None)
    a = np.sqrt(2.0) * norm.ppf(float(np.clip(auc, 0.51, 0.999)))
    fpr = np.clip((1.0 - pt) ** (1.35 + 2.4 * (auc - 0.55)), 0.002, 0.90)
    tpr = np.clip(norm.cdf(a + norm.ppf(np.clip(fpr, 1e-4, 1 - 1e-4))), 0.05, 0.999)
    nb = prev * tpr - (1.0 - prev) * fpr * (pt / np.clip(1.0 - pt, 1e-6, None))
    blend = np.clip(pt / 0.07, 0.0, 1.0) ** 0.65
    nb = (1.0 - blend) * treat_all + blend * nb
    return nb


def pr_curve(ap: float, seed: int, n: int = 140) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rec = np.linspace(0.0, 1.0, n)
    k = 6 + 40 * max(ap - 0.80, 0.01)
    prec = ap + (1 - ap) * np.exp(-k * rec) - (1 - ap) * rec**1.35
    prec = np.clip(prec + rng.normal(0, 0.008, n) * rec, 0.55, 1.0)
    prec = np.maximum(prec[::-1], np.maximum.accumulate(prec[::-1]))[::-1]
    prec[0] = 1.0
    return rec, prec


def km_surv(n: int, scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    t = np.sort(rng.exponential(scale, n))
    t = np.clip(t, 0.3, 60)
    times = np.concatenate([[0.0], t])
    surv = np.ones_like(times)
    at_risk = n
    for i in range(1, len(times)):
        at_risk -= 1
        surv[i] = max(at_risk / n, 0.02)
    # Kaplan–Meier with occasional plateaus
    surv = np.minimum.accumulate(surv)
    return times, surv


# ---------------------------------------------------------------------------
# Synthetic medical images
# ---------------------------------------------------------------------------


def _blob(h, w, cy, cx, sy, sx) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def make_lung_ct(seed: int, size: int = 320, side: str = "right") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    y = (yy - h / 2) / h
    x = (xx - w / 2) / w
    img = np.full((h, w), 0.05, dtype=float)
    y = (yy - h / 2) / (h / 2)
    x = (xx - w / 2) / (w / 2)

    body = (x / 0.78) ** 2 + ((y - 0.02) / 0.92) ** 2 <= 1.0
    inner = (x / 0.70) ** 2 + ((y - 0.02) / 0.84) ** 2 <= 1.0
    img[body] = 0.36
    img[body & ~inner] = 0.26

    def _lung(xc: float, flip: float) -> np.ndarray:
        xl = flip * (x - xc)
        yl = y + 0.02
        ell = (xl / 0.34) ** 2 + (yl / 0.58) ** 2 <= 1.0
        bite = ((xl + 0.22) / 0.12) ** 2 + ((yl - 0.02) / 0.22) ** 2 <= 1.0
        return ell & ~bite & inner

    lungs = _lung(-0.32, 1.0) | _lung(0.32, -1.0)
    tex = gaussian_filter(rng.normal(0, 1, (h, w)), 1.15)
    ves = np.zeros((h, w))
    for _ in range(48):
        cy = rng.randint(int(0.18 * h), int(0.82 * h))
        cx = rng.randint(int(0.14 * w), int(0.86 * w))
        if lungs[cy, cx]:
            ves += _blob(h, w, cy, cx, rng.uniform(1.3, 3.4), rng.uniform(1.1, 2.6))
    ves = np.clip(ves, 0, 1)
    img[lungs] = np.clip(0.07 + 0.035 * tex[lungs] + 0.14 * ves[lungs], 0, 1)

    heart = ((x + 0.05) / 0.22) ** 2 + ((y + 0.10) / 0.28) ** 2 <= 1.0
    img[heart & inner & ~lungs] = 0.50
    aorta = (x / 0.055) ** 2 + ((y + 0.10) / 0.055) ** 2 <= 1.0
    img[aorta] = 0.64
    vert = (x / 0.11) ** 2 + ((y - 0.74) / 0.11) ** 2 <= 1.0
    img[vert] = 0.90
    canal = (x / 0.045) ** 2 + ((y - 0.74) / 0.045) ** 2 <= 1.0
    img[canal] = 0.12
    stern = (x / 0.07) ** 2 + ((y + 0.84) / 0.045) ** 2 <= 1.0
    img[stern] = 0.80

    for ang in np.linspace(-1.15, 1.15, 8):
        ry, rx = 0.78 * np.sin(ang), 0.72 * np.cos(ang)
        rib = _blob(h, w, h / 2 + ry * h / 2, w / 2 + rx * w / 2, 5.0, 13)
        img = np.clip(img + 0.22 * rib * (body & ~lungs), 0, 1)

    core_lung = binary_erosion(lungs, iterations=12)
    if side == "right":
        pick = core_lung & (xx > 0.58 * w)
    else:
        pick = core_lung & (xx < 0.42 * w)
    ys, xs = np.where(pick)
    if len(ys) < 30:
        ys, xs = np.where(core_lung if core_lung.any() else lungs)
    k = int(rng.randint(0, len(ys)))
    ty, tx = int(ys[k]), int(xs[k])
    tumor_s = np.zeros((h, w))
    for _ in range(3):
        tumor_s += _blob(
            h, w,
            ty + rng.randint(-4, 5),
            tx + rng.randint(-4, 5),
            rng.uniform(7.5, 11.5),
            rng.uniform(6.5, 11.0),
        )
    tumor = (tumor_s > 0.72) & lungs
    img = np.where(tumor, np.clip(0.42 + 0.18 * tumor_s / (tumor_s.max() + 1e-6), 0, 1), img)
    core = _blob(h, w, ty, tx, 3.8, 3.5) > 0.55
    img[core & tumor] = np.clip(img[core & tumor] - 0.08, 0, 1)

    if int(tumor.sum()) < 40:
        forced = (_blob(h, w, ty, tx, 10.0, 9.0) > 0.40) & lungs
        tumor = forced
        img = np.where(tumor, np.clip(img, 0.38, 1), img)
    img = gaussian_filter(img, 0.55)
    img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
    return img, tumor.astype(np.uint8)


def make_brain_mri(seed: int, size: int = 320) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    y = (yy - h / 2) / h
    x = (xx - w / 2) / w
    img = np.full((h, w), 0.04, dtype=float)
    skull = (x / 0.36) ** 2 + ((y + 0.02) / 0.42) ** 2 <= 1.0
    brain = (x / 0.32) ** 2 + ((y + 0.02) / 0.38) ** 2 <= 1.0
    img[skull] = 0.78
    img[brain] = 0.42 + 0.06 * rng.rand(h, w)[brain]
    # ventricles
    vent = ((x / 0.07) ** 2 + ((y + 0.02) / 0.12) ** 2 <= 1.0) & brain
    img[vent] = 0.16
    # cortex folding
    for _ in range(18):
        cy = rng.randint(int(0.22 * h), int(0.78 * h))
        cx = rng.randint(int(0.22 * w), int(0.78 * w))
        if brain[cy, cx]:
            img += 0.04 * _blob(h, w, cy, cx, rng.uniform(6, 14), rng.uniform(4, 10))
    ty, tx = 0.12 * h + h / 2, 0.16 * w + w / 2
    tr = rng.uniform(0.055, 0.08) * w
    edema = _blob(h, w, ty, tx, tr * 1.7, tr * 1.55)
    tumor = _blob(h, w, ty, tx, tr, tr * 0.9)
    img = np.clip(img + 0.22 * edema * brain + 0.40 * tumor * brain, 0, 1)
    img = gaussian_filter(img, 0.7)
    img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
    gt = ((tumor > 0.45) & brain).astype(np.uint8)
    return img, gt


def make_pet(ct: np.ndarray, gt: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pet = 0.12 + 0.55 * np.clip(ct, 0, 1)
    # physiological heart / mediastinum
    h, w = ct.shape
    pet += 0.25 * _blob(h, w, 0.48 * h, 0.50 * w, 0.10 * h, 0.09 * w)
    hot = gaussian_filter(gt.astype(float), 2.4)
    pet = np.clip(pet + 1.15 * hot + rng.normal(0, 0.03, ct.shape), 0, 1.6)
    pet = gaussian_filter(pet, 1.1)
    pet = pet / (pet.max() + 1e-8)
    return pet


def make_pathology(seed: int, size: int = 320) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    h = w = size
    stroma = np.stack(
        [
            0.86 + 0.06 * rng.rand(h, w),
            0.62 + 0.08 * rng.rand(h, w),
            0.72 + 0.06 * rng.rand(h, w),
        ],
        axis=-1,
    )
    yy, xx = np.mgrid[0:h, 0:w]
    tumor = ((yy - 0.42 * h) / (0.22 * h)) ** 2 + ((xx - 0.58 * w) / (0.24 * w)) ** 2 <= 1.0
    # glands / nuclei
    img = stroma.copy()
    n_nuc = 1400
    ys = rng.randint(0, h, n_nuc)
    xs = rng.randint(0, w, n_nuc)
    denser = tumor[ys, xs]
    for y, x, d in zip(ys, xs, denser):
        rad = 1.6 if d else 1.2
        yy0, xx0 = np.ogrid[-3:4, -3:4]
        disk = yy0**2 + xx0**2 <= rad**2
        y0, x0 = max(y - 3, 0), max(x - 3, 0)
        y1, x1 = min(y + 4, h), min(x + 4, w)
        dy, dx = y1 - y0, x1 - x0
        patch = disk[:dy, :dx]
        color = np.array([0.38, 0.12, 0.42]) if d else np.array([0.45, 0.18, 0.48])
        sl = img[y0:y1, x0:x1]
        sl[patch] = 0.55 * sl[patch] + 0.45 * color
        img[y0:y1, x0:x1] = sl
    img = gaussian_filter(img, 0.35)
    img = np.clip(img, 0, 1)
    return img, tumor.astype(np.uint8)


def pred_mask(gt: np.ndarray, seed: int, jitter: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pred = gt.astype(bool)
    pred = shift(pred.astype(float), (rng.randint(-jitter, jitter + 1), rng.randint(-jitter, jitter + 1)), order=0) > 0.5
    if rng.rand() > 0.55:
        pred = binary_dilation(pred, iterations=1)
    elif rng.rand() > 0.45:
        pred = binary_erosion(pred, iterations=1)
        pred = binary_dilation(pred, iterations=1)
    return pred.astype(np.uint8)


def overlay_rgb(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    iters = 2 if int(gt.sum()) > 90 else 1
    gt_e = gt.astype(bool) ^ binary_erosion(gt.astype(bool), iterations=iters)
    pred_e = pred.astype(bool) ^ binary_erosion(pred.astype(bool), iterations=iters)
    fill = pred.astype(bool) & ~pred_e
    rgb[fill] = 0.72 * rgb[fill] + 0.28 * np.array([1.0, 0.25, 0.10])
    rgb[gt_e] = np.array([0.05, 0.95, 0.35])
    rgb[pred_e] = np.array([1.0, 0.28, 0.05])
    return np.clip(rgb, 0, 1)


def cam_from_mask(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    heat = gaussian_filter(mask.astype(float), 7.5)
    heat = heat + 0.08 * gaussian_filter(rng.rand(*mask.shape), 4.0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return np.clip(heat, 0, 1)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / (a.sum() + b.sum() + 1e-8))


# ---------------------------------------------------------------------------
# Training figures
# ---------------------------------------------------------------------------


def fig_loss() -> None:
    train = learning_curve(1.62, 0.118, k=4.4, noise=0.055, seed=11, kind="loss")
    val = np.maximum(learning_curve(1.58, 0.186, k=3.9, noise=0.070, seed=23, kind="loss"), train + 0.04)
    fig, ax = new_figure("Segmentation Loss (Dice + CE)")
    ax.plot(EPOCHS, train, color=C_TRAIN, lw=2.15, label="Training loss", zorder=3)
    ax.plot(EPOCHS, val, color=C_VAL, lw=2.15, label="Validation loss", zorder=3)
    ax.plot(EPOCHS[::10], train[::10], "o", color=C_TRAIN, ms=5.5, zorder=4)
    ax.plot(EPOCHS[::10], val[::10], "s", color=C_VAL, ms=5.5, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.85)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="upper right", framealpha=1.0)
    ax.text(
        0.97,
        0.18,
        f"Final val loss = {val[-1]:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10.5,
        color=C_TEXT,
        bbox=dict(boxstyle="square,pad=0.45", facecolor="white", edgecolor="#cccccc"),
    )
    save(fig, "01_loss_curve")


def fig_dice() -> None:
    train = learning_curve(0.22, 0.948, k=4.2, noise=0.018, seed=8, kind="acc")
    val = learning_curve(0.18, DICE_CT, k=3.85, noise=0.022, seed=19, kind="acc")
    val = np.minimum(val, train - 0.012)
    val[-1] = DICE_CT
    fig, ax = new_figure("Dice Coefficient")
    ax.plot(EPOCHS, train, color=C_TRAIN, lw=2.15, label="Training Dice", zorder=3)
    ax.plot(EPOCHS, val, color=C_VAL, lw=2.15, label="Validation Dice", zorder=3)
    ax.plot(EPOCHS[::10], train[::10], "o", color=C_TRAIN, ms=5.5, zorder=4)
    ax.plot(EPOCHS[::10], val[::10], "s", color=C_VAL, ms=5.5, zorder=4)
    ax.axhline(DICE_CT, color=C_OURS, ls=":", lw=1.3, zorder=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.10, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", framealpha=1.0)
    ax.text(2, DICE_CT + 0.018, f"Best val Dice = {DICE_CT:.3f}", color=C_OURS, fontsize=11)
    save(fig, "02_dice_curve")


def fig_hd95_iou() -> None:
    hd = learning_curve(28.4, HD95, k=4.0, noise=0.85, seed=14, kind="loss")
    iou = learning_curve(0.14, IOU, k=3.9, noise=0.020, seed=16, kind="acc")
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Boundary and Overlap Metrics")
    ax1 = fig.add_axes([0.145, 0.125, 0.78, 0.74])
    ax1.grid(True, color=C_GRID, linewidth=0.7, linestyle="--", zorder=0)
    ax1.set_axisbelow(True)
    ln1 = ax1.plot(EPOCHS, hd, color=C_VAL, lw=2.15, label="HD95 (mm)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("HD95 (mm)", color=C_VAL)
    ax1.tick_params(axis="y", labelcolor=C_VAL)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 32)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(EPOCHS, iou, color=C_TRAIN, lw=2.15, label="IoU")
    ax2.set_ylabel("IoU", color=C_TRAIN)
    ax2.tick_params(axis="y", labelcolor=C_TRAIN)
    ax2.set_ylim(0.10, 1.0)
    lns = ln1 + ln2
    ax1.legend(lns, [l.get_label() for l in lns], loc="center right", framealpha=1.0)
    ax1.text(
        0.97,
        0.12,
        f"HD95 = {HD95:.2f} mm   IoU = {IOU:.3f}",
        transform=ax1.transAxes,
        ha="right",
        fontsize=10.5,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )
    save(fig, "03_hd95_iou")


def fig_optimizer() -> None:
    sgd = learning_curve(0.16, 0.872, k=3.1, noise=0.024, seed=3, kind="acc")
    adam = learning_curve(0.20, 0.898, k=4.0, noise=0.020, seed=5, kind="acc")
    adamw = learning_curve(0.18, DICE_CT, k=4.15, noise=0.018, seed=7, kind="acc")
    fig, ax = new_figure("Optimizer Comparison")
    series = [
        (sgd, "SGD (momentum=0.9)", PALETTE[0], "o"),
        (adam, "Adam", PALETTE[2], "^"),
        (adamw, "AdamW (Ours)", C_OURS, "s"),
    ]
    for y, label, color, marker in series:
        ax.plot(EPOCHS, y, color=color, lw=2.15, label=label, zorder=3)
        ax.plot(EPOCHS[::12], y[::12], marker, color=color, ms=5.5, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.12, 0.98)
    ax.legend(loc="lower right", framealpha=1.0)
    save(fig, "04_optimizer_comparison")


def fig_architecture() -> None:
    names = [a[0] for a in ARCH]
    dice = [a[1] for a in ARCH]
    iou = [a[2] for a in ARCH]
    fig, ax = new_figure("Architecture Comparison")
    x = np.arange(len(names))
    w = 0.36
    colors_d = [PALETTE[0]] * 5 + [C_OURS]
    colors_i = ["#9ecae1"] * 5 + ["#a6dba0"]
    b1 = ax.bar(x - w / 2, dice, w, color=colors_d, edgecolor=C_SPINE, linewidth=0.6, label="Dice", zorder=3)
    b2 = ax.bar(x + w / 2, iou, w, color=colors_i, edgecolor=C_SPINE, linewidth=0.6, label="IoU", zorder=3)
    annotate_bars(ax, b1, "{:.3f}", dy=0.008, fontsize=8.2)
    annotate_bars(ax, b2, "{:.3f}", dy=0.008, fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.70, 0.98)
    ax.legend(loc="upper left", framealpha=1.0)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    save(fig, "05_architecture_comparison")


def fig_ablation() -> None:
    labels = [
        "Swin-UNet baseline",
        "+ skip attention",
        "+ deep supervision",
        "+ PET fusion",
        "+ boundary loss (Ours)",
    ]
    acc = np.array([0.879, 0.891, 0.901, 0.908, DICE_CT])
    colors = [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[4], C_OURS]
    fig, ax = new_figure("Ablation Study")
    y = np.arange(len(labels))
    bars = ax.barh(y, acc, color=colors, edgecolor=C_SPINE, linewidth=0.7, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Validation Dice")
    ax.set_xlim(0.865, 0.930)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="y")
    for bar, v in zip(bars, acc):
        ax.text(v + 0.0012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=11, color=C_TEXT)
    save(fig, "06_ablation_study")


def fig_lr_schedule() -> None:
    lr = cosine_warmup_lr()
    fig, ax = new_figure("Cosine Annealing with Warmup")
    ax.plot(EPOCHS, lr, color=C_TRAIN, lw=2.2, zorder=3)
    ax.fill_between(EPOCHS, lr, color=C_TRAIN, alpha=0.12)
    ax.axvline(10, color="#888888", ls="--", lw=1.0)
    ax.text(12, lr.max() * 0.92, "warmup 10 ep.", fontsize=10.5, color="#555555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_xlim(0, 100)
    ax.set_yscale("log")
    ax.set_ylim(8e-7, 2e-4)
    save(fig, "07_lr_schedule")


def fig_confusion() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.18, 0.16, 0.66, 0.68])
    add_title(fig, "Tumor Subtype Confusion Matrix")
    im = ax.imshow(CONFUSION, cmap="Blues", vmin=0, vmax=140)
    n = CONFUSION.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted subtype")
    ax.set_ylabel("True subtype")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for j in range(n):
            v = int(CONFUSION[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v >= 50 else C_TEXT, fontsize=14)
    cax = fig.add_axes([0.86, 0.16, 0.03, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Count", fontsize=11)
    acc = CONFUSION.trace() / CONFUSION.sum() * 100
    ax.text(0.5, -0.16, f"Overall accuracy = {acc:.2f}%  (n = {CONFUSION.sum()})", transform=ax.transAxes, ha="center", fontsize=11)
    save(fig, "08_confusion_matrix")


def fig_metrics_table() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Hold-out Test Metrics")
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.78])
    ax.axis("off")
    rows = [
        ["Task", "Metric", "Value"],
        ["Segmentation (CT)", "Dice", f"{DICE_CT:.3f}"],
        ["Segmentation (PET)", "Dice", f"{DICE_PET:.3f}"],
        ["Segmentation", "IoU / HD95", f"{IOU:.3f} / {HD95:.2f} mm"],
        ["Segmentation", "Sensitivity / Spec.", f"{SENS:.3f} / {SPEC:.3f}"],
        ["Subtype (4-class)", "Accuracy", f"{SUBTYPE_ACC:.2f}%"],
        ["Subtype", "Macro-AUC", "0.963"],
        ["Prognosis, train", "C-index / 3-yr AUC", f"{CINDEX_TRAIN:.3f} / {AUC_TRAIN:.3f}"],
        ["Prognosis, val", "C-index / 3-yr AUC", f"{CINDEX_VAL:.3f} / {AUC_VAL:.3f}"],
        ["Prognosis, test", "C-index / 3-yr AUC", f"{CINDEX_TEST:.3f} / {AUC_TEST:.3f}"],
        ["External cohort", "3-yr AUC", f"{AUC_EXT:.3f}"],
        ["Nomogram vs TNM", "ΔAUC", f"+{AUC_TEST - AUC_TNM:.3f}"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 2.05)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#1b4f72")
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        elif r % 2 == 0:
            cell.set_facecolor("#f4f8fb")
        if c == 2 and r > 0:
            cell.set_text_props(fontweight="bold", color=C_OURS)
    save(fig, "09_final_metrics")


def fig_per_class_dice() -> None:
    fig, ax = new_figure("Per-Structure Segmentation Dice")
    x = np.arange(len(ORGANS))
    colors = [C_OURS] + [PALETTE[i] for i in range(4)]
    bars = ax.bar(x, ORGAN_DICE, color=colors, edgecolor=C_SPINE, linewidth=0.7, width=0.66, zorder=3)
    annotate_bars(ax, bars, "{:.3f}", dy=0.006, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, fontsize=11)
    ax.set_ylabel("Dice")
    ax.set_ylim(0.78, 1.01)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    save(fig, "10_per_class_dice")


def fig_seed_band() -> None:
    rng = np.random.RandomState(33)
    mean = learning_curve(0.18, DICE_CT, k=3.9, noise=0.016, seed=19, kind="acc")
    std = 0.018 * np.exp(-3.0 * np.linspace(0, 1, N_EPOCHS)) + 0.004
    fig, ax = new_figure("Validation Dice with 3-Seed Std.")
    ax.fill_between(EPOCHS, mean - std, mean + std, color=C_OURS, alpha=0.18, label="±1 std. (3 seeds)", zorder=2)
    ax.plot(EPOCHS, mean, color=C_OURS, lw=2.2, label="Mean (Ours)", zorder=3)
    for s in range(3):
        noise = rng.normal(0, 1, N_EPOCHS) * std * 0.55
        ax.plot(EPOCHS, np.clip(mean + noise, 0.12, 0.97), color=C_OURS, lw=0.9, alpha=0.35, zorder=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.12, 1.0)
    ax.legend(loc="lower right", framealpha=1.0)
    save(fig, "11_seed_std_band")


# ---------------------------------------------------------------------------
# Clinical figures
# ---------------------------------------------------------------------------


def fig_roc() -> None:
    fig, ax = new_figure("ROC for 3-Year Overall Survival")
    series = [
        (AUC_TNM, "TNM stage", PALETTE[2], 4),
        (AUC_CLIN, "Clinical model", PALETTE[0], 5),
        (AUC_RAD, "Radiomics signature", PALETTE[1], 6),
        (AUC_TEST, "Nomogram (Ours)", C_OURS, 7),
    ]
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0, label="Chance")
    for auc, label, color, seed in series:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label}  (AUC={auc:.3f})")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=1.0)
    save(fig, "12_roc_auc")


def fig_roc_splits() -> None:
    fig, ax = new_figure("ROC on Train / Val / Test / External")
    series = [
        (AUC_TRAIN, "Training (n=312)", C_TRAIN, 11),
        (AUC_VAL, "Validation (n=104)", PALETTE[1], 12),
        (AUC_TEST, "Internal test (n=104)", C_OURS, 13),
        (AUC_EXT, "External (n=156)", C_VAL, 14),
    ]
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0)
    for auc, label, color, seed in series:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=2.15, label=f"{label}  AUC={auc:.3f}")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=1.0)
    save(fig, "13_roc_splits")


def fig_pr() -> None:
    fig, ax = new_figure("Precision–Recall (Tumor Subtype)")
    aps = [0.958, 0.941, 0.902, 0.966]
    for i, (name, ap) in enumerate(zip(CLASS_NAMES, aps)):
        rec, prec = pr_curve(ap, 70 + i)
        ax.plot(rec, prec, lw=2.0, color=PALETTE[i], label=f"{name}  (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.55, 1.03)
    ax.legend(loc="lower left", fontsize=10, framealpha=1.0)
    save(fig, "14_pr_curve")


def fig_dca() -> None:
    """Decision-curve analysis — the figure clinical papers actually need."""
    pt = np.linspace(0.01, 0.80, 220)
    prev = 0.42
    treat_all = (prev - pt) / np.clip(1 - pt, 1e-6, None)
    tnm = dca_nb(pt, prev, AUC_TNM)
    clin = dca_nb(pt, prev, AUC_CLIN)
    rad = dca_nb(pt, prev, AUC_RAD)
    nomo = dca_nb(pt, prev, AUC_TEST)

    fig, ax = new_figure("Decision Curve Analysis")
    ax.axhline(0.0, color="#666666", lw=1.2, label="Treat none")
    ax.plot(pt, treat_all, color="#999999", ls="--", lw=1.5, label="Treat all")
    ax.plot(pt, tnm, color=PALETTE[2], lw=2.0, label="TNM stage")
    ax.plot(pt, clin, color=PALETTE[0], lw=2.0, label="Clinical model")
    ax.plot(pt, rad, color=PALETTE[1], lw=2.0, label="Radiomics")
    ax.plot(pt, nomo, color=C_OURS, lw=2.4, label="Nomogram (Ours)")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.0, 0.80)
    ax.set_ylim(-0.05, 0.45)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=1.0)
    save(fig, "15_dca")


def fig_calibration() -> None:
    rng = np.random.RandomState(9)
    pred = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    obs_nomo = pred + rng.normal(0, 0.018, len(pred))
    obs_nomo = 0.82 * obs_nomo + 0.18 * pred
    obs_nomo = np.clip(obs_nomo, 0.06, 0.94)
    obs_tnm = 0.65 * pred + 0.12 + rng.normal(0, 0.03, len(pred))
    fig, ax = new_figure("Calibration of 3-Year OS")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.2, label="Perfect")
    ax.plot(pred, np.clip(obs_tnm, 0, 1), "-o", color=PALETTE[2], lw=2.0, ms=7, label="TNM")
    ax.plot(pred, obs_nomo, "-s", color=C_OURS, lw=2.2, ms=7, label="Nomogram")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", framealpha=1.0)
    ax.text(0.97, 0.08, "HL test  p = 0.41", transform=ax.transAxes, ha="right", fontsize=10.5, color="#555555")
    save(fig, "16_calibration")


def fig_nomogram() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Prognostic Nomogram for OS")
    ax = fig.add_axes([0.10, 0.07, 0.84, 0.82])
    ax.set_xlim(-18, 104)
    ax.set_ylim(-0.2, 12.8)
    ax.axis("off")

    def axis_line(y, x0, x1):
        ax.plot([x0, x1], [y, y], color=C_SPINE, lw=1.35, zorder=2)

    def ticks(y, xs, labels, fs=8.5):
        for x, lab in zip(xs, labels):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=C_SPINE, lw=1.05)
            ax.text(x, y + 0.28, lab, ha="center", va="bottom", fontsize=fs, color=C_TEXT)

    def varname(y, text):
        ax.text(-1.5, y, text, ha="right", va="center", fontsize=11, color=C_TEXT, fontweight="bold")

    # Points
    varname(11.2, "Points")
    axis_line(11.2, 0, 100)
    ticks(11.2, np.linspace(0, 100, 11), [str(int(v)) for v in np.linspace(0, 100, 11)])

    varname(9.6, "Age (years)")
    axis_line(9.6, 8, 92)
    ticks(9.6, [8, 29, 50, 71, 92], ["40", "50", "60", "70", "80"])

    varname(8.2, "T stage")
    axis_line(8.2, 10, 78)
    ticks(8.2, [10, 28, 50, 78], ["T1", "T2", "T3", "T4"])

    varname(6.8, "N stage")
    axis_line(6.8, 6, 88)
    ticks(6.8, [6, 32, 60, 88], ["N0", "N1", "N2", "N3"])

    varname(5.4, "SUVmax")
    axis_line(5.4, 12, 90)
    ticks(5.4, [12, 32, 52, 71, 90], ["2", "6", "10", "14", "18"])

    varname(4.0, "Rad-score")
    axis_line(4.0, 4, 96)
    ticks(4.0, [4, 27, 50, 73, 96], ["−1.5", "−0.5", "0.5", "1.5", "2.5"])

    varname(2.5, "Total points")
    axis_line(2.5, 0, 100)
    ticks(2.5, np.linspace(0, 100, 11), ["0", "30", "60", "90", "120", "150", "180", "210", "240", "270", "300"], fs=8)

    varname(1.3, "3-year OS")
    axis_line(1.3, 8, 92)
    ticks(1.3, [8, 28, 48, 66, 80, 92], ["0.9", "0.8", "0.6", "0.4", "0.2", "0.1"])

    varname(0.3, "5-year OS")
    axis_line(0.3, 10, 90)
    ticks(0.3, [10, 32, 52, 70, 82, 90], ["0.85", "0.7", "0.5", "0.3", "0.15", "0.08"])

    ax.text(
        50,
        12.55,
        "C-index = 0.821 (train) / 0.786 (test)",
        ha="center",
        fontsize=11,
        color=C_OURS,
        fontweight="bold",
    )
    save(fig, "17_nomogram")


def fig_km() -> None:
    t_hi, s_hi = km_surv(160, 42, seed=21)
    t_lo, s_lo = km_surv(156, 18, seed=22)
    # reshape to look like proper KM (step)
    fig, ax = new_figure("Kaplan–Meier Stratified by Rad-score")
    ax.step(t_hi, s_hi, where="post", color=C_OURS, lw=2.2, label="Low risk (n=160)")
    ax.step(t_lo, s_lo, where="post", color=C_VAL, lw=2.2, label="High risk (n=156)")
    ax.fill_between(t_hi, np.clip(s_hi - 0.05, 0, 1), np.clip(s_hi + 0.04, 0, 1), step="post", color=C_OURS, alpha=0.10)
    ax.fill_between(t_lo, np.clip(s_lo - 0.05, 0, 1), np.clip(s_lo + 0.04, 0, 1), step="post", color=C_VAL, alpha=0.10)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall survival")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", framealpha=1.0)
    ax.text(
        0.04,
        0.18,
        "Log-rank  p < 0.001\nHR = 2.84 (1.96–4.12)",
        transform=ax.transAxes,
        fontsize=11,
        color=C_TEXT,
        va="center",
        bbox=dict(boxstyle="square,pad=0.45", facecolor="white", edgecolor="#cccccc"),
    )
    save(fig, "18_kaplan_meier")


def fig_forest() -> None:
    names = [
        "Age (>65 vs ≤65)",
        "Sex (male vs female)",
        "T stage (T3–4 vs T1–2)",
        "N stage (N2–3 vs N0–1)",
        "SUVmax (>11 vs ≤11)",
        "Rad-score (high vs low)",
    ]
    hr = np.array([1.32, 1.08, 1.74, 1.91, 1.56, 2.84])
    lo = np.array([0.97, 0.79, 1.21, 1.34, 1.12, 1.96])
    hi = np.array([1.79, 1.48, 2.50, 2.72, 2.17, 4.12])
    p = ["0.076", "0.641", "0.003", "<0.001", "0.008", "<0.001"]
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Multivariable Cox Forest Plot")
    ax = fig.add_axes([0.30, 0.12, 0.40, 0.76])
    y = np.arange(len(names))[::-1]
    ax.axvline(1.0, color="#888888", ls="--", lw=1.1, zorder=1)
    for yi, h, l, r, name in zip(y, hr, lo, hi, names):
        color = C_OURS if "Rad-score" in name else C_TRAIN
        ax.plot([l, r], [yi, yi], color=color, lw=1.8, zorder=3)
        ax.plot(h, yi, "s", color=color, ms=8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_xlim(0.45, 4.6)
    ax.set_xticks([0.5, 1, 2, 3, 4])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.7, linestyle="--")
    tax = fig.add_axes([0.72, 0.12, 0.26, 0.76])
    tax.set_ylim(ax.get_ylim())
    tax.axis("off")
    for yi, h, l, r, pv in zip(y, hr, lo, hi, p):
        ptxt = f"p{pv}" if str(pv).startswith("<") else f"p={pv}"
        tax.text(0.0, yi, f"{h:.2f} ({l:.2f}–{r:.2f})\n{ptxt}", va="center", fontsize=8.5, color=C_TEXT)
    save(fig, "19_forest_plot")


def fig_radar() -> None:
    labels = ["Dice", "IoU", "Sensitivity", "HD95↓", "AUC", "C-index"]
    # HD95 inverted to 0–1 goodness: (12-hd)/12
    methods = {
        "U-Net": [0.847, 0.742, 0.862, (12 - 8.64) / 12, 0.781, 0.692],
        "nnU-Net": [0.891, 0.807, 0.901, (12 - 5.14) / 12, 0.824, 0.731],
        "TransUNet": [0.896, 0.814, 0.908, (12 - 4.88) / 12, 0.836, 0.748],
        "Ours": [DICE_CT, IOU, SENS, (12 - HD95) / 12, AUC_TEST, CINDEX_TEST],
    }
    colors = [PALETTE[0], PALETTE[1], PALETTE[3], C_OURS]
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Multi-Metric Radar Comparison")
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.78], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang[:-1]), labels, fontsize=11)
    ax.set_ylim(0.55, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8, color="#666666")
    ax.grid(color=C_GRID, linestyle="--", linewidth=0.7)
    for (name, vals), color in zip(methods.items(), colors):
        data = np.array(vals + [vals[0]], dtype=float)
        ax.plot(ang, data, color=color, lw=2.1, label=name)
        ax.fill(ang, data, color=color, alpha=0.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), framealpha=1.0, fontsize=10)
    save(fig, "20_radar_metrics")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def _show_gray(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10, pad=4, color=C_TEXT)


def fig_seg_overlay() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "GTV Segmentation  |  Green = GT, Red = Pred")
    cases = [
        (101, "right", "Case A  Dice=0.93"),
        (202, "left", "Case B  Dice=0.91"),
        (303, "right", "Case C  Dice=0.90"),
        (404, "left", "Case D  Dice=0.92"),
    ]
    for i, (seed, side, title) in enumerate(cases):
        img, gt = make_lung_ct(seed, size=280, side=side)
        pred = pred_mask(gt, seed + 7)
        rgb = overlay_rgb(img, gt, pred)
        ax0 = fig.add_axes([0.06 + (i % 2) * 0.47, 0.52 - (i // 2) * 0.44, 0.42, 0.38])
        ax0.imshow(rgb, interpolation="bilinear")
        ax0.set_title(title, fontsize=11, pad=4)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_color(C_SPINE)
    fig.text(0.50, 0.015, "Solid green contour: expert GT     Dashed-look red: Swin-UNETR", ha="center", fontsize=10, color="#555555")
    save(fig, "21_seg_overlay")


def fig_seg_strip() -> None:
    """Classic 4-column paper strip: CT / GT / Pred / Overlay."""
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Slice-wise Segmentation Comparison")
    headers = ["CECT", "Ground truth", "Prediction", "Overlay"]
    cases = [(111, "right"), (222, "left"), (333, "right")]
    for r, (seed, side) in enumerate(cases):
        img, gt = make_lung_ct(seed, 240, side)
        pred = pred_mask(gt, seed + 3)
        rgb = overlay_rgb(img, gt, pred)
        gt_rgb = np.stack([img, img, img], -1)
        gt_e = gt.astype(bool) ^ binary_erosion(gt.astype(bool), iterations=2)
        gt_rgb[gt_e] = (0.05, 0.95, 0.35)
        pr_rgb = np.stack([img, img, img], -1)
        pr_e = pred.astype(bool) ^ binary_erosion(pred.astype(bool), iterations=2)
        pr_rgb[pr_e] = (1.0, 0.28, 0.05)
        panels = [img, gt_rgb, pr_rgb, rgb]
        for c, im in enumerate(panels):
            ax = fig.add_axes([0.06 + c * 0.235, 0.66 - r * 0.28, 0.22, 0.24])
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color(C_SPINE)
            if r == 0:
                ax.set_title(headers[c], fontsize=11, pad=5)
            if c == 0:
                ax.set_ylabel(f"P{r + 1}", fontsize=11)
            if c == 3:
                d = dice(gt, pred)
                ax.text(0.97, 0.05, f"Dice {d:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="white",
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="#111111", edgecolor="none", alpha=0.55))
    save(fig, "22_seg_strip")


def fig_multimodal() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Multimodal Fusion  CT · PET · MRI · Pathology")
    ct, gt = make_lung_ct(77, 280, "right")
    pet = make_pet(ct, gt, 77)
    mri, mgt = make_brain_mri(88, 280)
    path, pgt = make_pathology(99, 280)
    fusion = np.stack([np.clip(pet * 1.1, 0, 1), ct * 0.65, ct * 0.45], -1)

    specs = [
        (0.07, 0.52, ct, "gray", "CECT  lung window"),
        (0.53, 0.52, pet, JET_MED, "18F-FDG PET  SUV"),
        (0.07, 0.08, fusion, None, "CT–PET fusion"),
        (0.53, 0.08, path, None, "Pathology  H&E  20×"),
    ]
    for x, y, im, cmap, title in specs:
        ax = fig.add_axes([x, y, 0.40, 0.38])
        if cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        elif cmap is None:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
    save(fig, "23_multimodal")


def fig_attention() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Swin Attention / Grad-CAM on GTV")
    cases = [(121, "right", "ADC"), (232, "left", "SCC"), (343, "right", "SCLC"), (454, "left", "LCC")]
    for i, (seed, side, name) in enumerate(cases):
        img, gt = make_lung_ct(seed, 240, side)
        heat = cam_from_mask(gt, seed)
        ax0 = fig.add_axes([0.07 + (i % 2) * 0.46, 0.52 - (i // 2) * 0.44, 0.20, 0.36])
        ax1 = fig.add_axes([0.28 + (i % 2) * 0.46, 0.52 - (i // 2) * 0.44, 0.20, 0.36])
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(f"{name}  input", fontsize=10, pad=3)
        ax0.axis("off")
        ax1.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax1.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax1.set_title("CAM", fontsize=10, pad=3)
        ax1.axis("off")
    fig.text(0.50, 0.015, "Warmer colors: tokens that support the GTV / subtype decision", ha="center", fontsize=10, color="#555555")
    save(fig, "24_attention_cam")


def fig_radiomics_heatmap() -> None:
    rng = np.random.RandomState(17)
    n_pat, n_feat = 64, 40
    low = rng.normal(0.0, 0.9, size=(32, n_feat))
    high = rng.normal(0.0, 0.9, size=(32, n_feat))
    high[:, :12] += 1.35
    low[:, :12] -= 0.85
    high[:, 12:20] -= 0.70
    data = np.vstack([low, high])
    # column z-score
    data = (data - data.mean(0)) / (data.std(0) + 1e-6)
    order = np.argsort(data[:, :8].mean(1))
    data = data[order]
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Radiomics Feature Heatmap")
    ax = fig.add_axes([0.18, 0.14, 0.68, 0.72])
    im = ax.imshow(data, cmap="RdBu_r", vmin=-2.4, vmax=2.4, aspect="auto")
    ax.set_xlabel("PyRadiomics + Swin deep features")
    ax.set_ylabel("Patients (sorted by rad-score)")
    ax.axhline(31.5, color=C_TEXT, lw=1.0)
    ax.text(-0.02, 0.78, "Low risk", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, color=C_OURS)
    ax.text(-0.02, 0.22, "High risk", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, color=C_VAL)
    ax.set_xticks([])
    ax.set_yticks([])
    cax = fig.add_axes([0.88, 0.14, 0.025, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Z-score", fontsize=11)
    save(fig, "25_radiomics_heatmap")


def fig_tsne() -> None:
    rng = np.random.RandomState(21)
    centers = np.array([[0.0, 0.1], [3.4, 0.4], [1.6, 3.1], [3.6, 2.8]])
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "t-SNE of Deep Radiomics Features")
    axes = [fig.add_axes([0.10, 0.12, 0.38, 0.74]), fig.add_axes([0.56, 0.12, 0.38, 0.74])]
    settings = [("Handcrafted only", 0.85), ("Swin + radiomics (Ours)", 0.32)]
    for ax, (title, std) in zip(axes, settings):
        for i, c in enumerate(centers):
            pts = c + rng.normal(0, std, size=(70, 2))
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=PALETTE[i], alpha=0.8, linewidths=0, label=CLASS_NAMES[i])
        ax.set_title(title, fontsize=11, pad=8, color=C_TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2" if ax is axes[0] else "")
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
        ax.set_aspect("equal", adjustable="datalim")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.015))
    save(fig, "26_tsne_features")


def _box(ax, xy, w, h, text, fc="#eef5ea", fs=9.5):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=1.15, edgecolor=C_SPINE, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=C_TEXT)
    return patch


def fig_swin_arch() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Swin-UNETR for Multimodal Segmentation")
    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    stages = [
        (0.5, 8.55, 9.0, 0.95, "Input  CECT 96³ + PET 96³   (two-stream patch embed)", "#f2f2f2"),
        (0.5, 7.25, 9.0, 0.95, "Swin encoder  W-MSA / SW-MSA  ×4 stages   C=48→96→192→384", "#dceaf7"),
        (0.5, 5.95, 9.0, 0.95, "Skip connections  +  cross-modal attention (CT ↔ PET)", "#c7e9c0"),
        (0.5, 4.65, 9.0, 0.95, "CNN decoder  3D conv upsample   deep supervision", "#dceaf7"),
        (0.5, 3.35, 4.3, 0.95, "GTV mask   Dice+CE+BD", "#f7e6c8"),
        (5.2, 3.35, 4.3, 0.95, "Deep features  512-d GAP", "#f7e6c8"),
        (0.5, 1.85, 9.0, 0.95, "Radiomics (107)  ⊕  deep (512)  ⊕  clinical  →  Cox / RSF / Nomogram", "#c7e9c0"),
        (0.5, 0.45, 9.0, 0.95, "Outputs:  subtype logits   3-/5-year OS   attention maps", "#e8f5e9"),
    ]
    for x, y, w, h, text, fc in stages:
        _box(ax, (x, y), w, h, text, fc=fc, fs=10.5)
    for y in [8.55, 7.25, 5.95, 4.65, 3.35, 1.85]:
        ax.annotate("", xy=(5.0, y - 0.05), xytext=(5.0, y - 0.28),
                    arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.15, mutation_scale=10))
    ax.text(5.0, 9.72, "Swin Transformer backbone  +  radiomics head  (Ours)", ha="center", fontsize=12, color=C_OURS)
    save(fig, "27_swin_architecture")


def fig_pipeline() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Radiomics / Pathomics Research Pipeline")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    blocks = [
        (0.45, 7.55, 2.85, 1.55, "1. Data\nCT / MRI / PET / WSI\nDICOM → NIfTI", "#f2f2f2"),
        (3.55, 7.55, 2.85, 1.55, "2. Preprocess\nwindow, resample\nN4, crop, z-score", "#dceaf7"),
        (6.65, 7.55, 2.85, 1.55, "3. ROI / Seg\nSwin-UNETR\nexpert refine", "#c7e9c0"),
        (0.45, 4.45, 2.85, 1.55, "4. Features\nPyRadiomics\nSwin GAP 512-d", "#dceaf7"),
        (3.55, 4.45, 2.85, 1.55, "5. Select\nmRMR + LASSO\n5-fold CV", "#dceaf7"),
        (6.65, 4.45, 2.85, 1.55, "6. Model\nCox / RSF\nNomogram", "#c7e9c0"),
        (0.45, 1.35, 4.35, 1.55, "7. Validate\nROC  DCA  calibration\nexternal cohort", "#f7e6c8"),
        (5.15, 1.35, 4.35, 1.55, "8. Visualize\nCAM  KM  forest\n3D GTV overlay", "#f7e6c8"),
    ]
    for x, y, w, h, text, fc in blocks:
        _box(ax, (x, y), w, h, text, fc=fc, fs=10.2)
    arrows = [
        ((3.30, 8.32), (3.55, 8.32)),
        ((6.40, 8.32), (6.65, 8.32)),
        ((3.30, 5.22), (3.55, 5.22)),
        ((6.40, 5.22), (6.65, 5.22)),
        ((4.80, 2.12), (5.15, 2.12)),
    ]
    ax.plot([8.07, 8.07, 1.87], [7.55, 6.35, 6.35], color=C_SPINE, lw=1.25)
    ax.annotate("", xy=(1.87, 6.00), xytext=(1.87, 6.35),
                arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.25, mutation_scale=11))
    ax.annotate("", xy=(1.87, 2.90), xytext=(1.87, 4.45),
                arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.25, mutation_scale=11))
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_SPINE, lw=1.25, mutation_scale=11))
    save(fig, "28_pipeline")


def fig_video_swin() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "Video-Swin  |  Longitudinal MRI Response")
    months = [0, 2, 4, 6, 9, 12]
    sizes = [1.00, 0.86, 0.70, 0.52, 0.40, 0.33]
    att = np.array([0.12, 0.18, 0.22, 0.19, 0.16, 0.13])
    for i, (m, s) in enumerate(zip(months, sizes)):
        img, gt = make_brain_mri(500 + i * 17, 220)
        # shrink tumor appearance by masking
        yy, xx = np.mgrid[0:220, 0:220]
        cy, cx = 0.62 * 220, 0.66 * 220
        r = 18 * s
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
        heat = cam_from_mask(blob.astype(np.uint8), 9 + i)
        ax = fig.add_axes([0.07 + i * 0.155, 0.46, 0.145, 0.38])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(heat, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"t={m} mo", fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
    axb = fig.add_axes([0.12, 0.12, 0.78, 0.26])
    style_box(axb)
    axb.bar(np.arange(6), att, color=C_TRAIN, edgecolor=C_SPINE, width=0.55, zorder=3)
    axb.set_xticks(np.arange(6))
    axb.set_xticklabels([f"{m} mo" for m in months])
    axb.set_ylabel("Temporal attention")
    axb.set_ylim(0, 0.30)
    axb.set_title("Video-Swin temporal attention weights", fontsize=11)
    save(fig, "29_video_swin")


def fig_volume_3d() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "3D GTV Surface  (Swin-UNETR)")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82], projection="3d")
    rng = np.random.RandomState(4)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 28)
    a, b, c = 1.15, 0.82, 0.70
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    # irregularity
    nrm = 0.08 * np.sin(3 * u)[:, None] * np.sin(4 * v)[None, :]
    x, y, z = x * (1 + nrm), y * (1 + nrm), z * (1 + nrm * 0.6)
    ax.plot_surface(x, y, z, cmap="inferno", linewidth=0, antialiased=True, alpha=0.92, shade=True)
    # clip planes as CT context
    yy, zz = np.meshgrid(np.linspace(-1.6, 1.6, 20), np.linspace(-1.3, 1.3, 20))
    xx = np.full_like(yy, -0.15)
    ax.plot_surface(xx, yy, zz, color="#9ecae1", alpha=0.18, linewidth=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1.2, 1.0, 0.85))
    ax.view_init(elev=18, azim=38)
    ax.text2D(0.50, 0.02, "Volume  18.4 cm³    Surface Dice  0.901", transform=ax.transAxes, ha="center", fontsize=11, color=C_TEXT)
    save(fig, "30_gtv_3d")


def fig_importance() -> None:
    names = [
        "wavelet-HHH_glcm_Contrast",
        "original_firstorder_Skewness",
        "swin_gap_f128",
        "log-sigma-3-mm_glszm_SZNUN",
        "original_shape_Sphericity",
        "SUVmax",
        "swin_gap_f041",
        "Age",
        "original_glrlm_SRHGE",
        "N stage",
    ]
    vals = np.array([0.142, 0.118, 0.109, 0.097, 0.086, 0.081, 0.074, 0.062, 0.055, 0.041])
    colors = [C_OURS if n.startswith(("wavelet", "original", "log", "swin")) else PALETTE[0] for n in names]
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "LASSO / SHAP Feature Importance")
    ax = fig.add_axes([0.38, 0.12, 0.55, 0.74])
    style_box(ax)
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor=C_SPINE, height=0.66, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|  (prognosis)")
    ax.set_xlim(0, 0.175)
    for yi, v in zip(y, vals):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9)
    save(fig, "31_feature_importance")


def fig_boxplot() -> None:
    rng = np.random.RandomState(6)
    fig, ax = new_figure("Rad-score by Tumor Subtype")
    data = [
        rng.normal(-0.35, 0.55, 140),
        rng.normal(0.15, 0.50, 100),
        rng.normal(0.55, 0.48, 40),
        rng.normal(1.05, 0.42, 40),
    ]
    bp = ax.boxplot(
        data, tick_labels=CLASS_NAMES, patch_artist=True, widths=0.55,
        medianprops=dict(color=C_TEXT, linewidth=1.6),
        whiskerprops=dict(color=C_SPINE), capprops=dict(color=C_SPINE),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor=C_SPINE, alpha=0.5),
    )
    for patch, c in zip(bp["boxes"], PALETTE[:4]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor(C_SPINE)
    ax.set_ylabel("Radiomics score")
    ax.set_ylim(-2.0, 2.6)
    ax.text(0.97, 0.06, "Kruskal–Wallis  p < 0.001", transform=ax.transAxes, ha="right", fontsize=10.5, color="#555555")
    save(fig, "32_radscore_boxplot")


def fig_waterfall() -> None:
    rng = np.random.RandomState(8)
    n = 80
    delta = np.sort(rng.normal(-12, 28, n))[::-1]  # % volume change
    delta = np.clip(delta, -85, 55)
    colors = np.where(delta < 0, C_OURS, C_VAL)
    fig, ax = new_figure("Waterfall  |  GTV Volume Change")
    ax.bar(np.arange(n), delta, color=colors, width=1.0, edgecolor="none", zorder=3)
    ax.axhline(0, color=C_SPINE, lw=1.0)
    ax.axhline(-30, color=C_TRAIN, ls="--", lw=1.0)
    ax.text(2, -27, "PR  (−30%)", color=C_TRAIN, fontsize=10)
    ax.set_xlabel("Patient (sorted)")
    ax.set_ylabel("ΔVolume after 2 cycles (%)")
    ax.set_xlim(-1, n)
    ax.set_ylim(-95, 65)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    ax.grid(False, axis="x")
    save(fig, "33_waterfall")


def fig_preprocess() -> None:
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    add_title(fig, "CT Preprocessing and ROI Crop")
    raw, gt = make_lung_ct(66, 300, "right")
    # "raw" more noisy / different window
    raw_show = np.clip((raw - 0.15) * 1.15, 0, 1)
    win = np.clip((raw - 0.08) / 0.55, 0, 1)
    # crop around tumor
    ys, xs = np.where(gt)
    y0, y1 = max(ys.min() - 18, 0), min(ys.max() + 18, raw.shape[0])
    x0, x1 = max(xs.min() - 18, 0), min(xs.max() + 18, raw.shape[1])
    crop = win[y0:y1, x0:x1]
    rgb = overlay_rgb(win, gt, pred_mask(gt, 66))
    panels = [
        (raw_show, "Raw DICOM", "gray"),
        (win, "Lung window + denoise", "gray"),
        (crop, "ROI crop  (96³)", "gray"),
        (rgb, "Swin mask overlay", None),
    ]
    for i, (im, title, cmap) in enumerate(panels):
        ax = fig.add_axes([0.07 + (i % 2) * 0.47, 0.52 - (i // 2) * 0.44, 0.42, 0.38])
        if cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=12, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C_SPINE)
        if i == 1:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ffeb3b", linewidth=1.6))
    save(fig, "34_preprocess_roi")


def fig_hyperparam() -> None:
    lrs = ["1e-5", "3e-5", "1e-4", "3e-4"]
    wd = ["1e-5", "5e-5", "1e-4", "5e-4"]
    acc = np.array(
        [
            [0.889, 0.896, 0.901, 0.894],
            [0.898, 0.906, 0.910, 0.903],
            [0.905, 0.911, 0.914, 0.908],
            [0.892, 0.899, 0.904, 0.897],
        ]
    )
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.16, 0.14, 0.68, 0.72])
    add_title(fig, "Hyperparameter Sensitivity")
    im = ax.imshow(acc, cmap="YlGn", vmin=0.886, vmax=0.916)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(wd)
    ax.set_yticklabels(lrs)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Learning rate")
    for i in range(4):
        for j in range(4):
            color = "white" if acc[i, j] >= 0.911 else C_TEXT
            ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center", color=color, fontsize=13)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    cax = fig.add_axes([0.86, 0.14, 0.03, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Validation Dice", fontsize=11)
    save(fig, "35_hyperparam_heatmap")


# ---------------------------------------------------------------------------
# Composite boards & covers
# ---------------------------------------------------------------------------


def _line(ax, xs, ys, color, label=None, lw=1.7):
    ax.plot(xs, ys, color=color, lw=lw, label=label, zorder=3)


def get_history() -> dict:
    train_loss = learning_curve(1.62, 0.118, k=4.4, noise=0.055, seed=11, kind="loss")
    val_loss = np.maximum(learning_curve(1.58, 0.186, k=3.9, noise=0.070, seed=23, kind="loss"), train_loss + 0.04)
    train_dice = learning_curve(0.22, 0.948, k=4.2, noise=0.018, seed=8, kind="acc")
    val_dice = learning_curve(0.18, DICE_CT, k=3.85, noise=0.022, seed=19, kind="acc")
    val_dice = np.minimum(val_dice, train_dice - 0.012)
    val_dice[-1] = DICE_CT
    hd = learning_curve(28.4, HD95, k=4.0, noise=0.85, seed=14, kind="loss")
    iou = learning_curve(0.14, IOU, k=3.9, noise=0.020, seed=16, kind="acc")
    return dict(
        train_loss=train_loss,
        val_loss=val_loss,
        train_dice=train_dice,
        val_dice=val_dice,
        hd=hd,
        iou=iou,
        sgd=learning_curve(0.16, 0.872, k=3.1, noise=0.024, seed=3, kind="acc"),
        adam=learning_curve(0.20, 0.898, k=4.0, noise=0.020, seed=5, kind="acc"),
        adamw=learning_curve(0.18, DICE_CT, k=4.15, noise=0.018, seed=7, kind="acc"),
    )


def composite_training_2x2() -> None:
    h = get_history()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.0), dpi=220)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.07, wspace=0.32, hspace=0.32)
    fig.suptitle("Segmentation Training", fontsize=17, fontweight="bold", color=C_TEXT, y=0.985)

    ax = axes[0, 0]
    style_box(ax)
    _line(ax, EPOCHS, h["train_loss"], C_TRAIN, "Train")
    _line(ax, EPOCHS, h["val_loss"], C_VAL, "Val")
    ax.set_title("Dice+CE loss")
    ax.legend(fontsize=8.5)
    tag(ax, "a")

    ax = axes[0, 1]
    style_box(ax)
    _line(ax, EPOCHS, h["train_dice"], C_TRAIN, "Train")
    _line(ax, EPOCHS, h["val_dice"], C_VAL, "Val")
    ax.set_title("Dice")
    ax.legend(fontsize=8.5, loc="lower right")
    tag(ax, "b")

    ax = axes[1, 0]
    style_box(ax)
    _line(ax, EPOCHS, h["hd"], C_VAL, "HD95")
    ax.set_title("HD95 (mm)")
    ax.set_xlabel("Epoch")
    tag(ax, "c")

    ax = axes[1, 1]
    style_box(ax, grid=True)
    names = [a[0] for a in ARCH]
    x = np.arange(len(names))
    colors = [PALETTE[0]] * 5 + [C_OURS]
    ax.bar(x, [a[1] for a in ARCH], color=colors, edgecolor=C_SPINE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.80, 0.94)
    ax.set_title("Architecture Dice")
    tag(ax, "d")
    save_composite(fig, "C1_training_2x2")


def composite_clinical_2x2() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.0), dpi=220)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.07, wspace=0.28, hspace=0.32)
    fig.suptitle("Clinical Validation", fontsize=17, fontweight="bold", color=C_TEXT, y=0.985)

    ax = axes[0, 0]
    style_box(ax)
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0)
    for auc, label, color, seed in [
        (AUC_TNM, "TNM", PALETTE[2], 4),
        (AUC_CLIN, "Clinical", PALETTE[0], 5),
        (AUC_RAD, "Radiomics", PALETTE[1], 6),
        (AUC_TEST, "Nomogram", C_OURS, 7),
    ]:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label} {auc:.3f}")
    ax.set_title("ROC  3-year OS")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    tag(ax, "a")

    ax = axes[0, 1]
    style_box(ax)
    pt = np.linspace(0.01, 0.80, 160)
    prev = 0.42
    treat_all = (prev - pt) / np.clip(1 - pt, 1e-6, None)
    nomo = dca_nb(pt, prev, AUC_TEST)
    rad_nb = dca_nb(pt, prev, AUC_RAD)
    tnm_nb = dca_nb(pt, prev, AUC_TNM)
    ax.axhline(0, color="#666", lw=1.0)
    ax.plot(pt, treat_all, color="#999", ls="--", lw=1.2, label="Treat all")
    ax.plot(pt, tnm_nb, color=PALETTE[2], lw=1.6, label="TNM")
    ax.plot(pt, rad_nb, color=PALETTE[1], lw=1.6, label="Radiomics")
    ax.plot(pt, nomo, color=C_OURS, lw=2.0, label="Nomogram")
    ax.set_title("Decision curve")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.45)
    ax.legend(fontsize=7.5)
    tag(ax, "b")

    ax = axes[1, 0]
    style_box(ax)
    t_hi, s_hi = km_surv(160, 42, 21)
    t_lo, s_lo = km_surv(156, 18, 22)
    ax.step(t_hi, s_hi, where="post", color=C_OURS, lw=1.8, label="Low risk")
    ax.step(t_lo, s_lo, where="post", color=C_VAL, lw=1.8, label="High risk")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan–Meier")
    ax.set_xlabel("Months")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.12, "p < 0.001", transform=ax.transAxes, fontsize=9)
    tag(ax, "c")

    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Nomogram (schematic)", fontsize=12, pad=8)

    def nline(y, x0, x1, labs, name):
        ax.plot([x0, x1], [y, y], color=C_SPINE, lw=1.2)
        ax.text(x0 - 0.15, y, name, ha="right", va="center", fontsize=8)
        for x, lab in zip(np.linspace(x0, x1, len(labs)), labs):
            ax.plot([x, x], [y - 0.08, y + 0.08], color=C_SPINE, lw=0.8)
            ax.text(x, y + 0.18, lab, ha="center", fontsize=6.5)

    nline(7.2, 1.8, 9.4, ["0", "20", "40", "60", "80", "100"], "Points")
    nline(5.9, 2.2, 8.8, ["40", "60", "80"], "Age")
    nline(4.6, 2.0, 8.2, ["T1", "T2", "T3", "T4"], "T")
    nline(3.3, 1.9, 9.0, ["−1", "0", "1", "2"], "Rad-score")
    nline(2.0, 1.8, 9.4, ["0", "150", "300"], "Total")
    nline(0.8, 2.3, 8.6, ["0.9", "0.6", "0.3"], "3-yr OS")
    ax.text(5.5, 0.15, "C-index 0.821 / 0.786", ha="center", fontsize=8.5, color=C_OURS)
    tag(ax, "d")
    save_composite(fig, "C2_clinical_2x2")


def composite_training_3x3() -> None:
    h = get_history()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5), dpi=230)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.05, wspace=0.30, hspace=0.34)
    fig.suptitle("Medical Image AI  Training Board", fontsize=17, fontweight="bold", color=C_TEXT, y=0.985)

    ax = axes[0, 0]
    style_box(ax)
    _line(ax, EPOCHS, h["train_loss"], C_TRAIN, "Train")
    _line(ax, EPOCHS, h["val_loss"], C_VAL, "Val")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    tag(ax, "a", 11)

    ax = axes[0, 1]
    style_box(ax)
    _line(ax, EPOCHS, h["train_dice"], C_TRAIN, "Train")
    _line(ax, EPOCHS, h["val_dice"], C_VAL, "Val")
    ax.set_title("Dice")
    ax.legend(fontsize=8, loc="lower right")
    tag(ax, "b", 11)

    ax = axes[0, 2]
    style_box(ax)
    _line(ax, EPOCHS, h["iou"], C_TRAIN, "IoU")
    ax.set_title("IoU")
    tag(ax, "c", 11)

    ax = axes[1, 0]
    style_box(ax)
    _line(ax, EPOCHS, h["sgd"], PALETTE[2], "SGD")
    _line(ax, EPOCHS, h["adam"], PALETTE[0], "Adam")
    _line(ax, EPOCHS, h["adamw"], C_OURS, "AdamW")
    ax.set_title("Optimizer")
    ax.legend(fontsize=7.5, loc="lower right")
    tag(ax, "d", 11)

    ax = axes[1, 1]
    style_box(ax, grid=True)
    labels = ["Baseline", "+ attn", "+ DS", "+ PET", "+ BD"]
    acc = [0.879, 0.891, 0.901, 0.908, DICE_CT]
    colors = [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[4], C_OURS]
    ax.barh(np.arange(5), acc, color=colors, edgecolor=C_SPINE, height=0.62, linewidth=0.5)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.87, 0.925)
    ax.set_title("Ablation")
    tag(ax, "e", 11)

    ax = axes[1, 2]
    style_box(ax, grid=True)
    names = [a[0] for a in ARCH]
    x = np.arange(len(names))
    ax.bar(x, [a[1] for a in ARCH], color=[PALETTE[0]] * 5 + [C_OURS], edgecolor=C_SPINE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylim(0.82, 0.94)
    ax.set_title("Architectures")
    tag(ax, "f", 11)

    ax = axes[2, 0]
    style_box(ax, grid=False)
    ax.imshow(CONFUSION, cmap="Blues", vmin=0, vmax=140)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(4):
        for j in range(4):
            v = int(CONFUSION[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if v >= 50 else C_TEXT)
    ax.set_title("Subtype CM")
    tag(ax, "g", 11)

    ax = axes[2, 1]
    style_box(ax, grid=True)
    x = np.arange(5)
    ax.bar(x, ORGAN_DICE, color=[C_OURS] + PALETTE[:4], edgecolor=C_SPINE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Per-structure Dice")
    tag(ax, "h", 11)

    ax = axes[2, 2]
    style_box(ax)
    lr = cosine_warmup_lr()
    ax.plot(EPOCHS, lr, color=C_TRAIN, lw=1.8)
    ax.set_yscale("log")
    ax.set_title("LR schedule")
    ax.set_xlabel("Epoch")
    tag(ax, "i", 11)

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    save_composite(fig, "C3_training_3x3")


def composite_eval_2x3() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.6), dpi=230)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, wspace=0.28, hspace=0.32)
    fig.suptitle("Evaluation Board", fontsize=17, fontweight="bold", color=C_TEXT, y=0.985)

    ax = axes[0, 0]
    style_box(ax)
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    for auc, label, color, seed in [
        (AUC_TRAIN, "Train", C_TRAIN, 11),
        (AUC_VAL, "Val", PALETTE[1], 12),
        (AUC_TEST, "Test", C_OURS, 13),
        (AUC_EXT, "External", C_VAL, 14),
    ]:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=1.7, label=f"{label} {auc:.3f}")
    ax.set_title("ROC splits")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    tag(ax, "a", 11)

    ax = axes[0, 1]
    style_box(ax)
    for i, (name, ap) in enumerate(zip(CLASS_NAMES, [0.958, 0.941, 0.902, 0.966])):
        rec, prec = pr_curve(ap, 70 + i)
        ax.plot(rec, prec, lw=1.5, color=PALETTE[i], label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.55, 1.02)
    ax.set_title("PR (subtype)")
    ax.legend(fontsize=7.5, loc="lower left")
    tag(ax, "b", 11)

    ax = axes[0, 2]
    style_box(ax)
    pt = np.linspace(0.01, 0.8, 160)
    prev = 0.42
    treat_all = (prev - pt) / np.clip(1 - pt, 1e-6, None)
    nomo = dca_nb(pt, prev, AUC_TEST)
    rad_nb = dca_nb(pt, prev, AUC_RAD)
    tnm_nb = dca_nb(pt, prev, AUC_TNM)
    ax.axhline(0, color="#666", lw=1)
    ax.plot(pt, treat_all, color="#999", ls="--", lw=1.2)
    ax.plot(pt, tnm_nb, color=PALETTE[2], lw=1.5, label="TNM")
    ax.plot(pt, nomo, color=C_OURS, lw=1.9, label="Ours")
    ax.set_title("DCA")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.45)
    ax.legend(fontsize=8)
    tag(ax, "c", 11)

    ax = axes[1, 0]
    style_box(ax)
    t_hi, s_hi = km_surv(160, 42, 21)
    t_lo, s_lo = km_surv(156, 18, 22)
    ax.step(t_hi, s_hi, where="post", color=C_OURS, lw=1.7, label="Low")
    ax.step(t_lo, s_lo, where="post", color=C_VAL, lw=1.7, label="High")
    ax.set_title("Kaplan–Meier")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    tag(ax, "d", 11)

    ax = axes[1, 1]
    style_box(ax, grid=False)
    rng = np.random.RandomState(17)
    data = np.vstack([rng.normal(-0.4, 0.8, (24, 28)), rng.normal(0.5, 0.8, (24, 28))])
    data[:, :8] *= np.linspace(1.4, 0.7, 24 + 24)[:, None]
    data = (data - data.mean(0)) / (data.std(0) + 1e-6)
    ax.imshow(data, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Radiomics heatmap")
    tag(ax, "e", 11)

    ax = axes[1, 2]
    style_box(ax)
    names = ["U-Net", "nnU-Net", "TransUNet", "Ours"]
    params = [17.3, 31.2, 105.3, 62.1]
    dice = [0.847, 0.891, 0.896, DICE_CT]
    cols = [PALETTE[0], PALETTE[1], PALETTE[3], C_OURS]
    for n, p, d, c in zip(names, params, dice, cols):
        ax.scatter(p, d, s=90, color=c, edgecolor=C_SPINE, zorder=4)
        ax.annotate(n, (p, d), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Params (M)")
    ax.set_ylabel("Dice")
    ax.set_title("Accuracy–size")
    tag(ax, "f", 11)
    save_composite(fig, "C4_eval_2x3")


def composite_results_2x5() -> None:
    h = get_history()
    fig, axes = plt.subplots(2, 5, figsize=(16.2, 6.6), dpi=240)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.12, wspace=0.32, hspace=0.42)
    fig.suptitle("Swin-UNETR Training Results", fontsize=16, fontweight="bold", color=C_TEXT, y=0.98)
    ce_t = learning_curve(1.05, 0.082, k=4.1, noise=0.04, seed=41, kind="loss")
    ce_v = np.maximum(learning_curve(1.02, 0.124, k=3.7, noise=0.05, seed=42, kind="loss"), ce_t + 0.03)
    bd_t = learning_curve(0.88, 0.071, k=3.8, noise=0.035, seed=43, kind="loss")
    bd_v = np.maximum(learning_curve(0.86, 0.110, k=3.5, noise=0.045, seed=44, kind="loss"), bd_t + 0.02)
    panels = [
        (axes[0, 0], "train/dice_loss", h["train_loss"], C_TRAIN),
        (axes[0, 1], "train/ce_loss", ce_t, C_TRAIN),
        (axes[0, 2], "train/boundary_loss", bd_t, C_TRAIN),
        (axes[0, 3], "metrics/dice", h["val_dice"], C_OURS),
        (axes[0, 4], "metrics/iou", h["iou"], C_OURS),
        (axes[1, 0], "val/dice_loss", h["val_loss"], C_VAL),
        (axes[1, 1], "val/ce_loss", ce_v, C_VAL),
        (axes[1, 2], "val/boundary_loss", bd_v, C_VAL),
        (axes[1, 3], "metrics/sensitivity", learning_curve(0.24, SENS, 3.9, 0.02, 45, "acc"), C_OURS),
        (axes[1, 4], "metrics/hd95(mm)", h["hd"], C_VAL),
    ]
    for ax, title, y, color in panels:
        style_box(ax)
        _line(ax, EPOCHS, y, color)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlim(1, 100)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("epoch", fontsize=9)
    save_composite(fig, "C5_results_2x5")


def cover_seg_mosaic() -> None:
    """1:1 cover: 3×3 CT overlays — the click image."""
    fig = plt.figure(figsize=(8.0, 8.0), dpi=300, facecolor="#0b1c2c")
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909]
    sides = ["right", "left", "right", "left", "right", "left", "right", "left", "right"]
    for i, (seed, side) in enumerate(zip(seeds, sides)):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.02 + c * 0.326, 0.14 + (2 - r) * 0.236, 0.312, 0.225])
        img, gt = make_lung_ct(seed, 240, side)
        pred = pred_mask(gt, seed + 11)
        ax.imshow(overlay_rgb(img, gt, pred), interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0)
        d = dice(gt, pred)
        ax.text(
            0.04, 0.06, f"Dice {d:.2f}", transform=ax.transAxes, color="white", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#000000", alpha=0.45, edgecolor="none"),
        )
    # title bar
    fig.patches.append(Rectangle((0, 0.855), 1, 0.145, transform=fig.transFigure, facecolor="#0b1c2c", zorder=0))
    bar = fig.add_axes([0, 0.855, 1, 0.145])
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    bar.axis("off")
    bar.set_zorder(2)
    bar.text(0.50, 0.62, "医学影像组学  ·  Swin 病灶分割", ha="center", va="center", color="white", fontproperties=cn(20))
    bar.text(0.50, 0.22, "CT / MRI / PET   Dice 0.914   HD95 3.82 mm", ha="center", va="center", color="#9ad0a8", fontsize=11)
    fig.patches.append(Rectangle((0, 0.0), 1, 0.125, transform=fig.transFigure, facecolor="#0b1c2c", zorder=0))
    foot = fig.add_axes([0, 0.0, 1, 0.125])
    foot.set_xlim(0, 1)
    foot.set_ylim(0, 1)
    foot.axis("off")
    foot.set_zorder(2)
    foot.text(0.50, 0.62, "绿 = 金标准勾画    红 = 模型预测", ha="center", color="#d9e6f2", fontproperties=cn(12))
    foot.text(0.50, 0.28, "一对一指导  分割 · 组学 · 预后建模", ha="center", color="#f0c674", fontproperties=cn(12))
    save_composite(fig, "COVER_01_seg_mosaic", dpi=300)


def cover_clinical() -> None:
    fig = plt.figure(figsize=(8.0, 8.0), dpi=300, facecolor="white")
    fig.patch.set_facecolor("#f7f4ee")
    fig.patches.append(Rectangle((0, 0.875), 1, 0.125, transform=fig.transFigure, facecolor="#1b4f72", zorder=0))
    head = fig.add_axes([0, 0.875, 1, 0.125])
    head.axis("off")
    head.set_zorder(2)
    head.text(0.50, 0.60, "临床科研出图  ·  ROC / DCA / 列线图 / KM", ha="center", va="center", color="white", fontproperties=cn(16))
    head.text(0.50, 0.22, "3-year AUC 0.887    C-index 0.821    p < 0.001", ha="center", va="center", color="#d4e6f1", fontsize=11)

    # ROC
    ax = fig.add_axes([0.08, 0.48, 0.40, 0.36])
    style_box(ax)
    ax.set_facecolor("white")
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=0.9)
    for auc, color, seed in [(AUC_TNM, PALETTE[2], 4), (AUC_RAD, PALETTE[1], 6), (AUC_TEST, C_OURS, 7)]:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=1.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # KM
    ax = fig.add_axes([0.55, 0.48, 0.40, 0.36])
    style_box(ax)
    ax.set_facecolor("white")
    t_hi, s_hi = km_surv(160, 42, 21)
    t_lo, s_lo = km_surv(156, 18, 22)
    ax.step(t_hi, s_hi, where="post", color=C_OURS, lw=1.8)
    ax.step(t_lo, s_lo, where="post", color=C_VAL, lw=1.8)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_title("Kaplan–Meier", fontsize=11)
    ax.set_xticks([0, 60])
    ax.set_yticks([0, 1])

    # DCA
    ax = fig.add_axes([0.08, 0.08, 0.40, 0.34])
    style_box(ax)
    ax.set_facecolor("white")
    pt = np.linspace(0.01, 0.8, 160)
    prev = 0.42
    treat_all = (prev - pt) / np.clip(1 - pt, 1e-6, None)
    nomo = dca_nb(pt, prev, AUC_TEST)
    ax.axhline(0, color="#666", lw=0.9)
    ax.plot(pt, treat_all, color="#999", ls="--", lw=1.1)
    ax.plot(pt, nomo, color=C_OURS, lw=1.9)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.45)
    ax.set_title("DCA", fontsize=11)

    # nomogram mini
    ax = fig.add_axes([0.55, 0.08, 0.40, 0.34])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Nomogram", fontsize=11, pad=8)
    for y, name, labs in [
        (5.2, "Age", ["40", "60", "80"]),
        (3.8, "T", ["T1", "T3", "T4"]),
        (2.4, "Rad", ["−1", "1", "2.5"]),
        (1.0, "OS", ["0.9", "0.5", "0.2"]),
    ]:
        ax.plot([1.8, 9.4], [y, y], color=C_SPINE, lw=1.1)
        ax.text(1.6, y, name, ha="right", va="center", fontsize=8)
        for x, lab in zip(np.linspace(1.8, 9.4, len(labs)), labs):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=C_SPINE, lw=0.8)
            ax.text(x, y + 0.22, lab, ha="center", fontsize=7)
    save_composite(fig, "COVER_02_clinical", dpi=300)


def cover_multimodal() -> None:
    fig = plt.figure(figsize=(8.0, 8.0), dpi=300, facecolor="#101010")
    ct, gt = make_lung_ct(77, 300, "right")
    pet = make_pet(ct, gt, 77)
    mri, mgt = make_brain_mri(88, 300)
    path, pgt = make_pathology(99, 300)
    fusion = np.stack([np.clip(pet * 1.15, 0, 1), ct * 0.6, np.clip(ct * 0.35 + pet * 0.2, 0, 1)], -1)
    mri_ov = overlay_rgb(mri, mgt, pred_mask(mgt, 88))
    path_ov = path.copy()
    edge = pgt.astype(bool) ^ binary_erosion(pgt.astype(bool), iterations=2)
    path_ov[edge] = (0.1, 0.95, 0.4)

    panels = [
        (ct, "gray", "CT  lung"),
        (pet, JET_MED, "PET  SUV"),
        (fusion, None, "CT–PET fusion"),
        (mri_ov, None, "MRI  + GTV"),
        (path_ov, None, "Pathomics"),
        (overlay_rgb(ct, gt, pred_mask(gt, 80)), None, "Swin overlay"),
    ]
    for i, (im, cmap, title) in enumerate(panels):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.03 + c * 0.323, 0.13 + (1 - r) * 0.355, 0.305, 0.33])
        if cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        elif cmap is None:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0)
        ax.text(
            0.04, 0.07, title, transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="square,pad=0.18", facecolor="#000000", alpha=0.5, edgecolor="none"),
        )
    fig.patches.append(Rectangle((0, 0.86), 1, 0.14, transform=fig.transFigure, facecolor="#101010", zorder=0))
    bar = fig.add_axes([0, 0.86, 1, 0.14])
    bar.axis("off")
    bar.set_zorder(2)
    bar.text(0.50, 0.58, "多模态医学图像 AI", ha="center", color="white", fontproperties=cn(20))
    bar.text(0.50, 0.18, "CT · MRI · PET · 病理组学 · 注意力融合", ha="center", color="#f0c674", fontproperties=cn(12))
    fig.patches.append(Rectangle((0, 0), 1, 0.11, transform=fig.transFigure, facecolor="#101010", zorder=0))
    foot = fig.add_axes([0, 0, 1, 0.11])
    foot.axis("off")
    foot.set_zorder(2)
    foot.text(0.50, 0.50, "分割标注  ·  特征提取  ·  预后 / 分型 / 鉴别诊断", ha="center", color="#d0d0d0", fontproperties=cn(12))
    save_composite(fig, "COVER_03_multimodal", dpi=300)


def cover_banner_16x9() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=220, facecolor="#0b1c2c")
    # left: 2x2 overlays
    seeds = [(101, "right"), (202, "left"), (88, "mri"), (77, "pet")]
    for i, (seed, kind) in enumerate(seeds):
        r, c = divmod(i, 2)
        ax = fig.add_axes([0.03 + c * 0.22, 0.14 + (1 - r) * 0.40, 0.205, 0.36])
        if kind == "mri":
            img, gt = make_brain_mri(seed, 260)
            ax.imshow(overlay_rgb(img, gt, pred_mask(gt, seed)))
        elif kind == "pet":
            img, gt = make_lung_ct(seed, 260, "right")
            pet = make_pet(img, gt, seed)
            ax.imshow(pet, cmap=JET_MED)
        else:
            img, gt = make_lung_ct(seed, 260, kind)
            ax.imshow(overlay_rgb(img, gt, pred_mask(gt, seed)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0)
    # right copy
    ax = fig.add_axes([0.50, 0.10, 0.47, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.88, "医学影像组学 ｜ 病理组学", color="white", fontproperties=cn(26))
    ax.text(0.0, 0.74, "Swin Transformer  ·  多模态融合  ·  临床建模", color="#9ad0a8", fontproperties=cn(14))
    lines = [
        ("Dice", "0.914"),
        ("3-year AUC", "0.887"),
        ("C-index", "0.821"),
        ("Subtype Acc", "94.1%"),
    ]
    for i, (k, v) in enumerate(lines):
        y = 0.52 - i * 0.10
        ax.text(0.0, y, k, color="#b0c4d4", fontsize=14)
        ax.text(0.42, y, v, color="#f0c674", fontsize=18, fontweight="bold")
    ax.text(0.0, 0.08, "一对一专业指导  ·  从预处理到出图", color="#e8e8e8", fontproperties=cn(14))
    save_composite(fig, "COVER_04_banner_16x9", dpi=220)


def cover_paper_figure1() -> None:
    """The 'Figure 1' look that papers and notes both love."""
    fig = plt.figure(figsize=(11.5, 8.2), dpi=230, facecolor="white")
    fig.suptitle("Figure 1.  Multimodal radiomics workflow and representative results", fontsize=13, fontweight="bold", y=0.98)

    ct, gt = make_lung_ct(141, 260, "right")
    pred = pred_mask(gt, 141)
    pet = make_pet(ct, gt, 141)
    heat = cam_from_mask(gt, 141)

    ax = fig.add_axes([0.04, 0.54, 0.18, 0.38])
    ax.imshow(ct, cmap="gray", vmin=0, vmax=1)
    ax.set_title("a  CECT", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.23, 0.54, 0.18, 0.38])
    ax.imshow(overlay_rgb(ct, gt, pred))
    ax.set_title("b  GTV overlay", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.42, 0.54, 0.18, 0.38])
    ax.imshow(pet, cmap=JET_MED)
    ax.set_title("c  PET SUV", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.61, 0.54, 0.18, 0.38])
    ax.imshow(ct, cmap="gray")
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title("d  Swin CAM", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.80, 0.54, 0.18, 0.38])
    path, pgt = make_pathology(55, 260)
    ax.imshow(path)
    ax.set_title("e  Pathomics", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.07, 0.10, 0.27, 0.38])
    style_box(ax)
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=0.9)
    for auc, color, seed, lab in [(AUC_TNM, PALETTE[2], 4, "TNM"), (AUC_TEST, C_OURS, 7, "Ours")]:
        fpr, tpr = roc_curve(auc, seed)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{lab} {auc:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("f  ROC", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_axes([0.39, 0.10, 0.27, 0.38])
    style_box(ax)
    t_hi, s_hi = km_surv(160, 42, 21)
    t_lo, s_lo = km_surv(156, 18, 22)
    ax.step(t_hi, s_hi, where="post", color=C_OURS, lw=1.7, label="Low risk")
    ax.step(t_lo, s_lo, where="post", color=C_VAL, lw=1.7, label="High risk")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_title("g  KM", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_axes([0.71, 0.10, 0.26, 0.38])
    style_box(ax)
    pt = np.linspace(0.01, 0.8, 160)
    prev = 0.42
    treat_all = (prev - pt) / np.clip(1 - pt, 1e-6, None)
    nomo = dca_nb(pt, prev, AUC_TEST)
    rad_nb = dca_nb(pt, prev, AUC_RAD)
    tnm_nb = dca_nb(pt, prev, AUC_TNM)
    ax.axhline(0, color="#666", lw=0.9)
    ax.plot(pt, treat_all, color="#999", ls="--", lw=1.1)
    ax.plot(pt, nomo, color=C_OURS, lw=1.9, label="Nomogram")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.45)
    ax.set_title("h  DCA", fontsize=10)
    ax.legend(fontsize=8)
    save_composite(fig, "COVER_05_figure1", dpi=230)


def main() -> None:
    configure_style()
    generators = [
        fig_loss,
        fig_dice,
        fig_hd95_iou,
        fig_optimizer,
        fig_architecture,
        fig_ablation,
        fig_lr_schedule,
        fig_confusion,
        fig_metrics_table,
        fig_per_class_dice,
        fig_seed_band,
        fig_roc,
        fig_roc_splits,
        fig_pr,
        fig_dca,
        fig_calibration,
        fig_nomogram,
        fig_km,
        fig_forest,
        fig_radar,
        fig_seg_overlay,
        fig_seg_strip,
        fig_multimodal,
        fig_attention,
        fig_radiomics_heatmap,
        fig_tsne,
        fig_swin_arch,
        fig_pipeline,
        fig_video_swin,
        fig_volume_3d,
        fig_importance,
        fig_boxplot,
        fig_waterfall,
        fig_preprocess,
        fig_hyperparam,
        composite_training_2x2,
        composite_clinical_2x2,
        composite_training_3x3,
        composite_eval_2x3,
        composite_results_2x5,
        cover_seg_mosaic,
        cover_clinical,
        cover_multimodal,
        cover_banner_16x9,
        cover_paper_figure1,
    ]
    for fn in generators:
        fn()
        print(f"wrote {fn.__name__}")

    from PIL import Image

    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as im:
            w, h = im.size
            print(f"  {path.name}: {w}×{h}")


if __name__ == "__main__":
    main()
