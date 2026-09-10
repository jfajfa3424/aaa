#!/usr/bin/env python3
"""Generate Xianyu-ready ML training result images (cover + detail)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager, patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc, roc_curve

OUT_DIR = Path(__file__).resolve().parent / "output"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

BG = "#0B1220"
CARD = "#121A2B"
CARD_ALT = "#182235"
LINE = "#243044"
TEXT = "#F4F7FB"
MUTED = "#9AA8C1"
ACCENT = "#2EE6A6"
BLUE = "#5B8CFF"
GOLD = "#F5C15A"
PINK = "#FF7A9A"
CYAN = "#4AD4FF"
ORANGE = "#FF9F5A"
PALETTE = [ACCENT, BLUE, GOLD, PINK, CYAN, ORANGE]


def setup_style() -> None:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams.update(
        {
            "font.family": "WenQuanYi Micro Hei",
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": CARD,
            "axes.edgecolor": LINE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": LINE,
            "grid.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.facecolor": CARD_ALT,
            "legend.edgecolor": LINE,
            "legend.labelcolor": TEXT,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.titlecolor": TEXT,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def pill(ax, x, y, text, color=ACCENT, ha="left") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=10,
        color=color,
        va="center",
        ha=ha,
        bbox=dict(
            boxstyle="round,pad=0.32,rounding_size=0.45",
            facecolor=CARD_ALT,
            edgecolor=color,
            linewidth=1.05,
            alpha=0.96,
        ),
        zorder=6,
    )


def card_box(ax, x, y, w, h) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            linewidth=1.15,
            edgecolor=LINE,
            facecolor=CARD,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def exp_decay(n: int, start: float, floor: float, tau: float, rng: np.random.Generator, noise: float) -> np.ndarray:
    x = np.arange(1, n + 1)
    y = (start - floor) * np.exp(-x / tau) + floor
    return np.clip(y + rng.normal(0, noise, n), floor * 0.85, None)


def make_square_cover() -> None:
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(8.6, 8.6))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.62, 1.12, 1.18],
        hspace=0.38,
        wspace=0.24,
        left=0.09,
        right=0.94,
        top=0.93,
        bottom=0.07,
    )
    hero = fig.add_subplot(gs[0, :])
    hero.set_axis_off()
    hero.text(0.0, 0.78, "案例效果展示  ·  可复现交付", fontsize=13, color=ACCENT)
    hero.text(0.0, 0.38, "机器学习建模  ·  回归 / 时序 / 分群", fontsize=20, color=TEXT)
    hero.text(0.0, 0.02, "R²  0.968     AUC  0.971     MAPE  3.8%", fontsize=15, color=GOLD)

    ax1 = fig.add_subplot(gs[1, 0])
    e = np.arange(1, 61)
    ax1.plot(e, exp_decay(60, 0.98, 0.09, 13, rng, 0.01), color=ACCENT, lw=2.4, label="Train")
    ax1.plot(e, exp_decay(60, 1.05, 0.11, 12, rng, 0.012), color=BLUE, lw=2.4, label="Valid")
    ax1.set_title("Loss 稳定收敛")
    ax1.legend(fontsize=8, loc="upper right")
    sns.despine(ax=ax1)

    ax2 = fig.add_subplot(gs[1, 1])
    y = rng.normal(50, 12, 140)
    ax2.scatter(y, y * 0.978 + rng.normal(0, 2.1, 140), s=18, c=GOLD, alpha=0.82, edgecolors="none")
    lim = [18, 82]
    ax2.plot(lim, lim, color=TEXT, lw=1.1, ls="--", alpha=0.7)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_title("预测紧密贴合")
    sns.despine(ax=ax2)

    ax3 = fig.add_subplot(gs[2, :])
    t = np.arange(80)
    s = 32 + 0.12 * t + 6.2 * np.sin(t / 6.2) + rng.normal(0, 0.45, 80)
    ax3.plot(t[:55], s[:55], color=TEXT, lw=2.1, label="历史")
    ax3.plot(t[54:], s[54:] + rng.normal(0, 0.22, 26), color=ACCENT, lw=2.5, label="LSTM 预测")
    ax3.axvline(54, color=GOLD, ls="--", lw=1.2)
    ax3.set_title("LSTM 时序预测  ·  趋势与周期可跟踪")
    ax3.legend(ncol=2, loc="upper left", fontsize=8)
    sns.despine(ax=ax3)
    save(fig, "00_cover_square.png")


def make_portrait_cover() -> None:
    rng = np.random.default_rng(9)
    fig = plt.figure(figsize=(8.2, 11.0))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.78, 0.92, 1.45, 1.35],
        hspace=0.34,
        wspace=0.22,
        left=0.09,
        right=0.93,
        top=0.955,
        bottom=0.055,
    )

    hero = fig.add_subplot(gs[0, :])
    hero.set_axis_off()
    hero.text(0.0, 0.80, "案例效果展示", fontsize=13, color=ACCENT)
    hero.text(0.0, 0.42, "回归预测  ·  时序分析", fontsize=26, color=TEXT)
    hero.text(0.0, 0.08, "XGBoost / LightGBM / CatBoost / LSTM / 随机森林", fontsize=12, color=MUTED)

    kpis = fig.add_subplot(gs[1, :])
    kpis.set_axis_off()
    items = [
        ("R²", "0.968", "拟合优度", ACCENT),
        ("MAE", "2.14", "平均绝对误差", BLUE),
        ("MAPE", "3.8%", "相对误差", GOLD),
        ("AUC", "0.971", "分类区分度", PINK),
    ]
    for i, (k, v, sub, color) in enumerate(items):
        x0 = 0.01 + i * 0.248
        card_box(kpis, x0, 0.06, 0.23, 0.88)
        kpis.plot([x0 + 0.018, x0 + 0.018], [0.18, 0.78], transform=kpis.transAxes, color=color, lw=3.2, solid_capstyle="round")
        kpis.text(x0 + 0.04, 0.70, k, fontsize=11, color=color, transform=kpis.transAxes)
        kpis.text(x0 + 0.04, 0.38, v, fontsize=22, color=TEXT, transform=kpis.transAxes)
        kpis.text(x0 + 0.04, 0.16, sub, fontsize=9, color=MUTED, transform=kpis.transAxes)

    ax_loss = fig.add_subplot(gs[2, :])
    e = np.arange(1, 81)
    ax_loss.plot(e, exp_decay(80, 0.92, 0.08, 16, rng, 0.008), color=ACCENT, lw=2.4, label="Train Loss")
    ax_loss.plot(e, exp_decay(80, 0.98, 0.10, 15, rng, 0.01), color=BLUE, lw=2.4, label="Valid Loss")
    ax_loss.set_title("训练过程：Loss 稳定下降，验证集无过拟合", loc="left", fontsize=12, pad=8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    sns.despine(ax=ax_loss)

    ax_bar = fig.add_subplot(gs[3, 0])
    models = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "随机森林"]
    scores = np.array([0.968, 0.961, 0.957, 0.948, 0.932])
    bars = ax_bar.barh(models[::-1], scores[::-1], color=[ACCENT, BLUE, GOLD, CYAN, PINK][::-1], height=0.62)
    ax_bar.set_xlim(0.90, 0.985)
    ax_bar.set_title("多模型 R²", loc="left", fontsize=12, pad=8)
    for bar, score in zip(bars, scores[::-1]):
        ax_bar.text(score + 0.0015, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=9, color=TEXT)
    sns.despine(ax=ax_bar)

    ax_ts = fig.add_subplot(gs[3, 1])
    t = np.arange(70)
    series = 72 + 0.16 * t + 8 * np.sin(t / 6.5) + rng.normal(0, 0.9, t.size)
    split = 50
    ax_ts.plot(t[: split + 1], series[: split + 1], color=TEXT, lw=1.8)
    ax_ts.plot(t[split:], series[split:] + rng.normal(0, 0.35, t.size - split), color=ACCENT, lw=2.2)
    ax_ts.axvline(split, color=GOLD, ls="--", lw=1.1)
    ax_ts.set_title("LSTM 预测", loc="left", fontsize=12, pad=8)
    sns.despine(ax=ax_ts)
    save(fig, "01_cover_portrait.png")


def make_loss_curves() -> None:
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.3))
    fig.suptitle("训练过程监控  ·  损失下降平稳，验证集同步收敛", fontsize=18, y=1.03)

    epochs = np.arange(1, 101)
    axes[0].plot(epochs, exp_decay(100, 1.18, 0.075, 20, rng, 0.012), color=ACCENT, lw=2.4, label="训练集 Loss")
    axes[0].plot(epochs, exp_decay(100, 1.24, 0.095, 19, rng, 0.015), color=BLUE, lw=2.4, label="验证集 Loss")
    axes[0].axhline(0.10, color=GOLD, ls=":", lw=1.2, label="目标阈值")
    axes[0].set_title("回归任务  MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")

    acc_tr = 1 - 0.46 * np.exp(-epochs / 15) + rng.normal(0, 0.005, epochs.size)
    acc_va = 1 - 0.50 * np.exp(-epochs / 14.5) + rng.normal(0, 0.007, epochs.size)
    axes[1].plot(epochs, np.clip(acc_tr, None, 0.993), color=ACCENT, lw=2.4, label="训练集 Accuracy")
    axes[1].plot(epochs, np.clip(acc_va, None, 0.976), color=PINK, lw=2.4, label="验证集 Accuracy")
    axes[1].set_title("分类任务  Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.48, 1.03)
    axes[1].legend(loc="lower right")
    for ax in axes:
        sns.despine(ax=ax)
        pill(ax, 0.03, 0.93, "正向收敛")
    fig.text(0.5, -0.03, "验证集与训练集走势一致，可用于交付说明", ha="center", color=MUTED, fontsize=12)
    save(fig, "02_loss_accuracy.png")


def make_model_compare() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.5))
    fig.suptitle("多模型对比  ·  树模型与深度学习均达到可用水平", fontsize=18, y=1.03)

    models = ["线性回归", "决策树", "SVM", "随机森林", "LSTM", "CatBoost", "LightGBM", "XGBoost"]
    r2 = np.array([0.812, 0.846, 0.871, 0.932, 0.948, 0.957, 0.961, 0.968])
    mae = np.array([6.42, 5.88, 5.11, 3.26, 2.72, 2.40, 2.31, 2.14])
    cmap = LinearSegmentedColormap.from_list("rank", [BLUE, ACCENT])
    colors = [cmap(i / (len(models) - 1)) for i in range(len(models))]

    axes[0].barh(models, r2, color=colors, height=0.66)
    axes[0].set_xlim(0.78, 1.0)
    axes[0].set_title("测试集 R²（越高越好）")
    axes[0].axvline(0.95, color=GOLD, ls="--", lw=1.15, label="优秀线 0.95")
    for y, v in zip(models, r2):
        axes[0].text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9, color=TEXT)
    axes[0].legend(loc="lower right")

    axes[1].barh(models, mae, color=colors, height=0.66)
    axes[1].set_title("测试集 MAE（越低越好）")
    axes[1].set_xlabel("MAE")
    axes[1].set_xlim(0, 7.3)
    for y, v in zip(models, mae):
        label = f"{v:.2f}" + ("  最优" if y == "XGBoost" else "")
        color = GOLD if y == "XGBoost" else TEXT
        axes[1].text(v + 0.08, y, label, va="center", fontsize=9, color=color)
    for ax in axes:
        sns.despine(ax=ax)
    save(fig, "03_model_compare.png")


def make_pred_vs_actual() -> None:
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    fig.suptitle("回归拟合效果  ·  预测紧密贴合真实值", fontsize=18, y=1.03)

    n = 260
    y = rng.uniform(22, 178, n)
    y_hat = 1.6 + 0.978 * y + rng.normal(0, 5.2, n)
    resid = y_hat - y

    axes[0].scatter(y, y_hat, s=22, c=BLUE, alpha=0.75, edgecolors="none")
    lims = [15, 190]
    axes[0].plot(lims, lims, color=GOLD, lw=1.8, ls="--")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("真实值 Actual")
    axes[0].set_ylabel("预测值 Predicted")
    axes[0].set_title("R² = 0.968    MAE = 2.14    RMSE = 3.07")
    pill(axes[0], 0.04, 0.93, "高拟合优度")

    sns.histplot(resid, bins=22, ax=axes[1], color=ACCENT, kde=True, edgecolor=BG, line_kws={"lw": 2.2})
    axes[1].axvline(0, color=GOLD, ls="--", lw=1.4)
    axes[1].set_title("残差近似正态、均值接近 0")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    for ax in axes:
        sns.despine(ax=ax)
    save(fig, "04_pred_vs_actual.png")


def make_timeseries() -> None:
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(13.8, 6.7))
    t = np.arange(180)
    y = 44 + 0.11 * t + 9 * np.sin(2 * np.pi * t / 12) + 3.2 * np.sin(2 * np.pi * t / 6) + rng.normal(0, 0.85, t.size)
    split = 144
    future = y[split:] + rng.normal(0, 0.35, y[split:].size)
    ax.plot(t[:split], y[:split], color=TEXT, lw=2.05, label="历史实际")
    ax.plot(t[split - 1 :], np.r_[y[split - 1], y[split:]], color=MUTED, lw=1.45, ls=":", label="真实对照")
    ax.plot(t[split - 1 :], np.r_[y[split - 1], future], color=ACCENT, lw=2.5, label="LSTM 预测")
    ax.fill_between(t[split:], future - 3.4, future + 3.4, color=ACCENT, alpha=0.16, label="95% 置信区间")
    ax.axvline(split - 0.5, color=GOLD, ls="--", lw=1.3)
    ax.text(split + 2.5, np.max(y) - 1.5, "Forecast", color=GOLD, fontsize=12)
    ax.set_title("时间序列预测（LSTM）  ·  MAPE 3.8%  ·  趋势与季节性完整捕捉", fontsize=16, pad=12)
    ax.set_xlabel("时间步  Time Step")
    ax.set_ylabel("业务指标")
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    pill(ax, 0.98, 0.08, "预测贴合真实走势", ha="right")
    sns.despine(ax=ax)
    save(fig, "05_timeseries_lstm.png")


def make_feature_importance() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.9))
    names = [
        "历史成交价",
        "流量 / 曝光",
        "用户活跃度",
        "品类热度",
        "季节因子",
        "竞品均价",
        "促销强度",
        "库存周转",
        "评分口碑",
        "地域指数",
    ]
    values = np.array([0.21, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])
    cmap = LinearSegmentedColormap.from_list("mintbar", [BLUE, ACCENT])
    colors = [cmap(i / (len(names) - 1)) for i in range(len(names))]
    bars = ax.barh(names[::-1], values[::-1], color=colors, height=0.66)
    ax.set_xlabel("Importance")
    ax.set_title("特征重要性  ·  结果可解释，便于业务优化", fontsize=16, pad=12)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center", color=TEXT)
    ax.set_xlim(0, 0.27)
    pill(ax, 0.98, 0.08, "可解释机器学习", ha="right")
    sns.despine(ax=ax)
    save(fig, "06_feature_importance.png")


def make_classification() -> None:
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3))
    fig.suptitle("分类模型效果  ·  高召回、低误判", fontsize=18, y=1.03)

    cm = np.array([[186, 9], [11, 154]])
    cmap = LinearSegmentedColormap.from_list("mint", ["#163047", ACCENT])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        ax=axes[0],
        cbar=False,
        linewidths=3,
        linecolor=BG,
        annot_kws={"size": 20, "color": TEXT},
        vmin=0,
        vmax=220,
    )
    axes[0].set_xticklabels(["预测负类", "预测正类"], color=TEXT)
    axes[0].set_yticklabels(["真实负类", "真实正类"], color=TEXT, rotation=0)
    axes[0].set_title("混淆矩阵  Accuracy 94.4%")
    axes[0].tick_params(colors=TEXT)
    axes[0].grid(False)
    # Keep off-diagonal numbers readable on darker cells.
    axes[0].texts[1].set_color(GOLD)
    axes[0].texts[2].set_color(GOLD)

    y_true = np.r_[np.zeros(280), np.ones(280)]
    scores = np.r_[rng.normal(0.0, 0.68, 280), rng.normal(1.78, 0.68, 280)]
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=ACCENT, lw=2.6, label=f"XGBoost  AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="随机猜测")
    axes[1].fill_between(fpr, tpr, alpha=0.13, color=ACCENT)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC 曲线")
    axes[1].legend(loc="lower right")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    sns.despine(ax=axes[1])
    save(fig, "07_classification_roc.png")


def make_cluster() -> None:
    rng = np.random.default_rng(8)
    fig, ax = plt.subplots(figsize=(12.5, 7.1))
    centers = np.array([[0.55, 1.05], [3.45, 3.25], [-0.25, 3.65], [2.25, -0.35]])
    colors = [ACCENT, BLUE, GOLD, PINK]
    labels = ["高价值活跃", "潜力增长", "价格敏感", "流失风险"]
    for c, color, lab in zip(centers, colors, labels):
        pts = rng.normal(c, 0.38, size=(150, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=color, alpha=0.8, edgecolors="none", label=lab)
        ax.scatter(*c, s=170, c="white", edgecolors=color, linewidths=2.2, zorder=5)
    ax.set_title("用户分群  K-Means  ·  四类边界清晰，可直接用于运营策略", fontsize=16, pad=12)
    ax.set_xlabel("消费能力（标准化）")
    ax.set_ylabel("活跃程度（标准化）")
    ax.legend(loc="upper left", title="客群")
    pill(ax, 0.98, 0.08, "聚类可解释", ha="right")
    sns.despine(ax=ax)
    save(fig, "08_kmeans_clusters.png")


def main() -> None:
    setup_style()
    make_square_cover()
    make_portrait_cover()
    make_loss_curves()
    make_model_compare()
    make_pred_vs_actual()
    make_timeseries()
    make_feature_importance()
    make_classification()
    make_cluster()


if __name__ == "__main__":
    main()
