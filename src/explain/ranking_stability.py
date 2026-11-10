
"""Analyze channel-ranking stability across 10 cross-validation folds."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT / "outputs" / "interpretability"
    / "cdl_mddnet_kfold10_baseline"
)

INPUT_FILE = OUTPUT_DIR / "all_folds_channel_rankings.csv"

N_FOLDS = 10
TOP_K_VALUES = (10, 20)


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {"fold", "channel", "score", "rank"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["fold"].nunique() != N_FOLDS:
        raise ValueError(
            f"Expected {N_FOLDS} folds, found {df['fold'].nunique()}"
        )

    if df.duplicated(["fold", "channel"]).any():
        raise ValueError("Duplicate channel entries found within a fold.")

    counts = df.groupby("fold")["channel"].nunique()

    if not (counts == 128).all():
        raise ValueError(
            f"Expected 128 channels per fold. Found:\n{counts}"
        )

    print("Validated 10 folds and 128 channels per fold.")

    # ---------------------------------------------------------
    # 1. Create fold x channel matrices
    # ---------------------------------------------------------

    score_matrix = df.pivot(
        index="fold",
        columns="channel",
        values="score",
    )

    rank_matrix = df.pivot(
        index="fold",
        columns="channel",
        values="rank",
    )

    # ---------------------------------------------------------
    # 2. Mean and standard deviation of channel importance
    # ---------------------------------------------------------

    mean_scores = score_matrix.mean(axis=0)
    std_scores = score_matrix.std(axis=0, ddof=1)

    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks = rank_matrix.std(axis=0, ddof=1)

    stability = pd.DataFrame({
        "channel": score_matrix.columns,
        "mean_score": mean_scores.values,
        "std_score": std_scores.values,
        "mean_rank": mean_ranks.values,
        "std_rank": std_ranks.values,
    })

    # ---------------------------------------------------------
    # 3. Top-k selection frequency
    # ---------------------------------------------------------

    for k in TOP_K_VALUES:
        top_k_counts = (
            rank_matrix.le(k)
            .sum(axis=0)
            .reindex(score_matrix.columns)
        )

        stability[f"top_{k}_count"] = top_k_counts.values
        stability[f"top_{k}_frequency"] = (
            top_k_counts.values / N_FOLDS
        )

    # Sort by mean importance.
    stability = stability.sort_values(
        "mean_score",
        ascending=False,
    ).reset_index(drop=True)

    stability.insert(
        0,
        "overall_rank",
        np.arange(1, len(stability) + 1),
    )

    stability.to_csv(
        OUTPUT_DIR / "ranking_stability.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Pairwise Spearman correlations between folds
    # ---------------------------------------------------------

    correlation_matrix = rank_matrix.T.corr(
        method="spearman"
    )

    correlation_matrix.to_csv(
        OUTPUT_DIR / "fold_spearman_correlation.csv"
    )

    upper_triangle = correlation_matrix.to_numpy()[
        np.triu_indices(N_FOLDS, k=1)
    ]

    mean_correlation = float(np.mean(upper_triangle))
    min_correlation = float(np.min(upper_triangle))
    max_correlation = float(np.max(upper_triangle))

    # ---------------------------------------------------------
    # 5. Export Member 6 handoff
    # ---------------------------------------------------------

    final_ranking = stability[
        ["channel", "mean_score"]
    ].rename(
        columns={"mean_score": "score"}
    )

    final_ranking.to_csv(
        OUTPUT_DIR / "channel_ranking.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Print results
    # ---------------------------------------------------------

    print("\n========== TOP 20 STABLE CHANNELS ==========")

    print(
        stability.head(20).to_string(index=False)
    )

    print("\n========== FOLD RANKING STABILITY ==========")

    print("Mean pairwise Spearman:", mean_correlation)
    print("Minimum Spearman:", min_correlation)
    print("Maximum Spearman:", max_correlation)

    print("\n========== TOP-10 FREQUENCY ==========")

    print(
        stability.sort_values(
            "top_10_frequency",
            ascending=False,
        )[
            ["channel", "mean_score", "top_10_count",
             "top_10_frequency"]
        ].head(20).to_string(index=False)
    )

    print("\nSaved:")
    print("  ranking_stability.csv")
    print("  fold_spearman_correlation.csv")
    print("  channel_ranking.csv")


if __name__ == "__main__":
    main()
