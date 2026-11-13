
"""Inspect model performance and channel ranking differences across folds."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "cdl_mddnet_kfold10_baseline"

RUN_DIR = ROOT / "runs" / EXPERIMENT
OUTPUT_DIR = (
    ROOT / "outputs" / "interpretability" / EXPERIMENT
)


def main():
    performance_rows = []
    top_channel_rows = []

    for fold in range(10):
        fold_dir = RUN_DIR / f"fold_{fold}"

        with open(fold_dir / "fold_meta.json") as f:
            meta = json.load(f)

        ranking = pd.read_csv(
            OUTPUT_DIR / f"fold_{fold}" / "channel_ranking.csv"
        )

        performance_rows.append({
            "fold": fold,
            "test_accuracy": meta["test_acc"],
            "test_loss": meta["test_loss"],
            "test_auc": meta.get("test_auc", np.nan),
            "best_epoch": meta.get("best_epoch", np.nan),
            "test_epochs": meta.get("n_test_epochs", np.nan),
        })

        top_channels = ranking.head(10)["channel"].tolist()

        for rank, channel in enumerate(top_channels, start=1):
            top_channel_rows.append({
                "fold": fold,
                "rank": rank,
                "channel": channel,
            })

    performance = pd.DataFrame(performance_rows)
    top_channels = pd.DataFrame(top_channel_rows)

    performance.to_csv(
        OUTPUT_DIR / "fold_performance_diagnostics.csv",
        index=False,
    )

    top_channels.to_csv(
        OUTPUT_DIR / "fold_top10_channels.csv",
        index=False,
    )

    print("\n========== FOLD PERFORMANCE ==========")
    print(performance.to_string(index=False))

    print("\n========== TOP 10 CHANNELS PER FOLD ==========")

    for fold in range(10):
        channels = top_channels[
            top_channels["fold"] == fold
        ]["channel"].tolist()

        print(f"Fold {fold}: {', '.join(channels)}")

    print("\n========== PERFORMANCE SUMMARY ==========")

    print(
        performance[
            ["test_accuracy", "test_loss", "test_auc"]
        ].agg(["mean", "std", "min", "max"]).to_string()
    )

    print("\nSaved diagnostic files.")


if __name__ == "__main__":
    main()
