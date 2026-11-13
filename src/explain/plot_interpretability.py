
"""Generate final interpretability visualizations for CDL-MDDNet."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT / "outputs" / "interpretability"
    / "cdl_mddnet_kfold10_baseline"
)

PLOT_DIR = RESULTS_DIR / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_band_ablation():
    """Mean loss change with standard deviation across folds."""

    df = pd.read_csv(
        RESULTS_DIR / "band_ablation_summary.csv"
    )

    df = df.sort_values("mean_loss_increase")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        df["band"],
        df["mean_loss_increase"],
        yerr=df["std_loss_increase"],
        capsize=5,
    )

    ax.axhline(0, linewidth=1, linestyle="--")

    ax.set_title("Frequency Band Ablation Across 10 Folds")
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Mean Change in Test Loss")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "band_ablation_mean_loss.png",
        dpi=300,
    )
    plt.close(fig)

    print("Saved band_ablation_mean_loss.png")


def plot_band_fold_heatmap():
    """Show loss increase for each band in each fold."""

    df = pd.read_csv(
        RESULTS_DIR / "band_ablation_all_folds.csv"
    )

    pivot = df.pivot(
        index="band",
        columns="fold",
        values="loss_increase",
    )

    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    pivot = pivot.reindex(band_order)

    fig, ax = plt.subplots(figsize=(11, 5))

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Cross-validation Fold")
    ax.set_ylabel("Frequency Band")
    ax.set_title("Frequency Band Ablation: Loss Change by Fold")

    fig.colorbar(
        image,
        ax=ax,
        label="Ablated Loss - Baseline Loss",
    )

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "band_ablation_fold_heatmap.png",
        dpi=300,
    )
    plt.close(fig)

    print("Saved band_ablation_fold_heatmap.png")


def plot_channel_stability():
    """Pairwise Spearman correlation between fold rankings."""

    df = pd.read_csv(
        RESULTS_DIR / "fold_spearman_correlation.csv",
        index_col=0,
    )

    matrix = df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Fold")
    ax.set_title("Channel Ranking Stability: Pairwise Spearman Correlation")

    fig.colorbar(
        image,
        ax=ax,
        label="Spearman Correlation",
    )

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "channel_ranking_stability.png",
        dpi=300,
    )
    plt.close(fig)

    print("Saved channel_ranking_stability.png")


def plot_top_channel_frequency():
    """Plot how frequently channels appear in the top 10."""

    df = pd.read_csv(
        RESULTS_DIR / "ranking_stability.csv"
    )

    df = df.sort_values(
        ["top_10_count", "mean_score"],
        ascending=[False, False],
    )

    top = df.head(20).sort_values("top_10_count")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top["channel"],
        top["top_10_count"],
    )

    ax.set_xlabel("Number of Folds in Which Channel Appears in Top 10")
    ax.set_ylabel("EEG Channel")
    ax.set_title("Top-10 Channel Selection Frequency Across 10 Folds")
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11))
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "top_channel_frequency.png",
        dpi=300,
    )
    plt.close(fig)

    print("Saved top_channel_frequency.png")


def main():
    print("Generating interpretability figures...")

    plot_band_ablation()
    plot_band_fold_heatmap()
    plot_channel_stability()
    plot_top_channel_frequency()

    print("\nAll figures saved in:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
