
"""Integrated Gradients explanations for CDL-MDDNet."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src.models.cdl_mddnet import CDLMDDNet
from src.data.normalize import apply_normalizer


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "cdl_mddnet_kfold10_baseline"
FOLD = 0

DATA_PATH = ROOT / "data" / "processed" / "modma_epochs.npz"
FOLD_DIR = ROOT / "runs" / EXPERIMENT / f"fold_{FOLD}"

OUTPUT_DIR = (
    ROOT / "outputs" / "interpretability" / EXPERIMENT / f"fold_{FOLD}"
)

DEVICE = torch.device("cpu")

BATCH_SIZE = 8
IG_STEPS = 64


def load_data():
    data = np.load(DATA_PATH, allow_pickle=False)

    return (
        data["X"],
        data["y"],
        data["subject_id"],
        data["ch_names"],
    )


def load_model_and_normalizer():
    stats_path = FOLD_DIR / "normalizer_stats.npz"

    with np.load(stats_path, allow_pickle=False) as ns:
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
        FOLD_DIR / "model.pt",
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, stats


def explain_fold():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y, subject_ids, ch_names = load_data()

    preds = pd.read_csv(FOLD_DIR / "preds.csv")
    epoch_indices = preds["epoch_idx"].to_numpy(dtype=np.int64)

    X_test = X[epoch_indices]
    y_test = y[epoch_indices]
    test_subject_ids = subject_ids[epoch_indices]

    print("Loading model and fold normalizer...")
    model, stats = load_model_and_normalizer()

    X_normalized = apply_normalizer(X_test, stats)

    inputs = torch.from_numpy(X_normalized).float()

    # Zero baseline in normalized input space.
    baseline = torch.zeros_like(inputs)

    ig = IntegratedGradients(model)

    all_attributions = []
    all_deltas = []

    total = len(inputs)

    print(f"Computing IG for {total} test epochs...")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = inputs[start:end]
        batch_baseline = baseline[start:end]

        attributions, delta = ig.attribute(
            batch,
            baselines=batch_baseline,
            target=1,
            n_steps=IG_STEPS,
            method="gausslegendre",
            internal_batch_size=BATCH_SIZE,
            return_convergence_delta=True,
        )

        all_attributions.append(attributions.detach().cpu().numpy())
        all_deltas.append(delta.detach().cpu().numpy())

        print(f"Processed {end}/{total} epochs")

    attributions = np.concatenate(all_attributions, axis=0)
    convergence_delta = np.concatenate(all_deltas, axis=0)

    # Shape: (epochs, channels, time points)
    assert attributions.shape == (total, 128, 1000)

    # Mean absolute attribution across epochs and time.
    channel_scores = np.mean(np.abs(attributions), axis=(0, 2))

    ranking = pd.DataFrame({
        "channel": ch_names.astype(str),
        "score": channel_scores,
    }).sort_values("score", ascending=False)

    ranking.to_csv(
        OUTPUT_DIR / "channel_ranking.csv",
        index=False,
    )

    np.savez_compressed(
        OUTPUT_DIR / "integrated_gradients.npz",
        attributions=attributions,
        epoch_indices=epoch_indices,
        y_true=y_test,
        subject_id=test_subject_ids,
        convergence_delta=convergence_delta,
        ch_names=ch_names,
    )

    print("\nIntegrated Gradients completed!")
    print("Attribution shape:", attributions.shape)

    print(
        "Mean absolute convergence delta:",
        float(np.mean(np.abs(convergence_delta))),
    )

    print("\nTop 15 channels:")
    print(ranking.head(15).to_string(index=False))

    print("\nOutput directory:", OUTPUT_DIR)


if __name__ == "__main__":
    explain_fold()
