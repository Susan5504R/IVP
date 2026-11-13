
"""Run Integrated Gradients channel attribution across all 10 K-fold splits."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src.models.cdl_mddnet import CDLMDDNet
from src.data.normalize import apply_normalizer


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "cdl_mddnet_kfold10_baseline"
DATA_PATH = ROOT / "data" / "processed" / "modma_epochs.npz"

OUTPUT_ROOT = (
    ROOT / "outputs" / "interpretability" / EXPERIMENT
)

DEVICE = torch.device("cpu")

N_FOLDS = 10
BATCH_SIZE = 8
IG_STEPS = 64


def load_dataset():
    data = np.load(DATA_PATH, allow_pickle=False)

    return (
        data["X"],
        data["y"],
        data["subject_id"],
        data["ch_names"].astype(str),
    )


def load_fold_model_and_stats(fold_dir):
    with np.load(
        fold_dir / "normalizer_stats.npz",
        allow_pickle=False,
    ) as ns:
        stats = {
            "mean": ns["mean"],
            "std": ns["std"],
            "eps": float(ns["eps"]),
            "n_channels": int(ns["n_channels"]),
        }

    model = CDLMDDNet(
        n_channels=stats["n_channels"],
        n_samples=1000,
        n_classes=2,
        n_subjects=0,
    ).to(DEVICE)

    state_dict = torch.load(
        fold_dir / "model.pt",
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, stats


def validate_fold_mapping(preds, meta, y, subject_ids, ch_names):
    indices = preds["epoch_idx"].to_numpy(dtype=np.int64)

    assert len(indices) > 0
    assert indices.min() >= 0
    assert indices.max() < len(y)
    assert len(np.unique(indices)) == len(indices)

    assert np.array_equal(
        y[indices],
        preds["y_true"].to_numpy(dtype=np.int64),
    )

    assert np.array_equal(
        subject_ids[indices],
        preds["subject_id"].to_numpy(dtype=np.int64),
    )

    assert set(np.unique(subject_ids[indices])) == set(
        meta["test_subjects"]
    )

    assert np.array_equal(
        ch_names,
        np.asarray(meta["ch_names"], dtype=str),
    )

    return indices


def explain_one_fold(fold, X, y, subject_ids, ch_names):
    print(f"\n{'=' * 55}")
    print(f"Starting fold {fold}")
    print(f"{'=' * 55}")

    fold_dir = (
        ROOT / "runs" / EXPERIMENT / f"fold_{fold}"
    )

    output_dir = OUTPUT_ROOT / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(fold_dir / "preds.csv")

    with open(fold_dir / "fold_meta.json", "r") as f:
        meta = json.load(f)

    epoch_indices = validate_fold_mapping(
        preds, meta, y, subject_ids, ch_names
    )

    X_test = X[epoch_indices]

    model, stats = load_fold_model_and_stats(fold_dir)

    X_normalized = apply_normalizer(X_test, stats)

    ig = IntegratedGradients(model)

    channel_score_sum = np.zeros(
        stats["n_channels"], dtype=np.float64
    )

    total_attribution_elements = 0
    convergence_deltas = []

    total_epochs = len(X_normalized)

    for start in range(0, total_epochs, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_epochs)

        batch = torch.from_numpy(
            X_normalized[start:end]
        ).float().to(DEVICE)

        baseline = torch.zeros_like(batch)

        attributions, delta = ig.attribute(
            batch,
            baselines=baseline,
            target=1,
            n_steps=IG_STEPS,
            method="gausslegendre",
            internal_batch_size=BATCH_SIZE,
            return_convergence_delta=True,
        )

        attr_np = attributions.detach().cpu().numpy()

        # Sum absolute attribution over epochs and time.
        channel_score_sum += np.abs(attr_np).sum(axis=(0, 2))

        total_attribution_elements += (
            attr_np.shape[0] * attr_np.shape[2]
        )

        convergence_deltas.append(
            delta.detach().cpu().numpy()
        )

        print(f"Fold {fold}: {end}/{total_epochs} epochs")

    channel_scores = (
        channel_score_sum / total_attribution_elements
    )

    ranking = pd.DataFrame({
        "channel": ch_names,
        "score": channel_scores,
    }).sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    ranking.to_csv(
        output_dir / "channel_ranking.csv",
        index=False,
    )

    mean_delta = float(
        np.mean(np.abs(np.concatenate(convergence_deltas)))
    )

    summary = {
        "fold": fold,
        "test_epochs": total_epochs,
        "test_subjects": sorted(
            np.unique(subject_ids[epoch_indices]).tolist()
        ),
        "ig_steps": IG_STEPS,
        "target_class": 1,
        "mean_absolute_convergence_delta": mean_delta,
    }

    with open(output_dir / "ig_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nFold {fold} completed.")
    print("Mean absolute convergence delta:", mean_delta)
    print("Top 5 channels:")
    print(ranking.head(5).to_string(index=False))

    return ranking


def main():
    print("Loading complete EEG dataset...")
    X, y, subject_ids, ch_names = load_dataset()

    all_rankings = []

    for fold in range(N_FOLDS):
        ranking = explain_one_fold(
            fold,
            X,
            y,
            subject_ids,
            ch_names,
        )

        ranking["fold"] = fold
        all_rankings.append(ranking)

    combined = pd.concat(
        all_rankings,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_ROOT / "all_folds_channel_rankings.csv",
        index=False,
    )

    print("\nAll 10 folds completed.")
    print(
        "Saved combined rankings:",
        OUTPUT_ROOT / "all_folds_channel_rankings.csv",
    )


if __name__ == "__main__":
    main()
