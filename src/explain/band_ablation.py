
"""Frequency-band ablation across all 10 CV folds for CDL-MDDNet."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

from src.models.cdl_mddnet import CDLMDDNet
from src.data.normalize import apply_normalizer


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "cdl_mddnet_kfold10_baseline"
N_FOLDS = 10

DATA_PATH = ROOT / "data" / "processed" / "modma_epochs.npz"
RUNS_DIR = ROOT / "runs" / EXPERIMENT

OUTPUT_DIR = ROOT / "outputs" / "interpretability" / EXPERIMENT

FS = 250
FILTER_ORDER = 4
BATCH_SIZE = 8

DEVICE = torch.device("cpu")

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def load_data():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        return (
            data["X"],
            data["y"],
            data["subject_id"],
        )


def load_model_and_normalizer(fold_dir):
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


def extract_band(signal, low, high):
    """Extract a frequency band from EEG shaped (N, C, T)."""

    sos = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    return sosfiltfilt(sos, signal, axis=-1)


def evaluate_model(model, signal, stats):
    normalized = apply_normalizer(signal, stats)

    x = torch.from_numpy(normalized).float().to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)

    return logits.cpu().numpy(), probabilities.cpu().numpy()


def calculate_metrics(logits, probabilities, labels):
    labels_tensor = torch.from_numpy(labels).long()
    logits_tensor = torch.from_numpy(logits).float()

    loss = F.cross_entropy(
        logits_tensor,
        labels_tensor,
    ).item()

    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == labels)

    return float(loss), float(accuracy)


def evaluate_in_batches(model, signal, stats):
    logits_list = []
    probabilities_list = []

    for start in range(0, len(signal), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(signal))

        logits, probabilities = evaluate_model(
            model,
            signal[start:end],
            stats,
        )

        logits_list.append(logits)
        probabilities_list.append(probabilities)

    return (
        np.concatenate(logits_list),
        np.concatenate(probabilities_list),
    )


def run_fold(fold, X, y, subject_ids):
    print(f"\n{'=' * 60}")
    print(f"PROCESSING FOLD {fold}")
    print(f"{'=' * 60}")

    fold_dir = RUNS_DIR / f"fold_{fold}"

    if not (fold_dir / "model.pt").exists():
        raise FileNotFoundError(f"Missing checkpoint: {fold_dir / 'model.pt'}")

    if not (fold_dir / "normalizer_stats.npz").exists():
        raise FileNotFoundError(f"Missing normalizer: {fold_dir}")

    preds = pd.read_csv(fold_dir / "preds.csv")

    epoch_indices = preds["epoch_idx"].to_numpy(dtype=np.int64)

    if len(np.unique(epoch_indices)) != len(epoch_indices):
        raise ValueError(f"Duplicate epoch indices in fold {fold}")

    if np.any(epoch_indices < 0) or np.any(epoch_indices >= len(X)):
        raise ValueError(f"Invalid epoch index in fold {fold}")

    X_test = X[epoch_indices]
    y_test = y[epoch_indices]
    test_subject_ids = subject_ids[epoch_indices]

    # Verify that the prediction file maps to the same test labels.
    if not np.array_equal(
        preds["y_true"].to_numpy(dtype=np.int64),
        y_test,
    ):
        raise ValueError(f"Label mismatch in fold {fold}")

    # Verify subject IDs when available.
    if "subject_id" in preds.columns:
        if not np.array_equal(
            preds["subject_id"].to_numpy(dtype=np.int64),
            test_subject_ids,
        ):
            raise ValueError(f"Subject mapping mismatch in fold {fold}")

    print("Test epochs:", len(X_test))
    print("Test subjects:", len(np.unique(test_subject_ids)))

    model, stats = load_model_and_normalizer(fold_dir)

    print("Evaluating original EEG...")

    baseline_logits, baseline_probs = evaluate_in_batches(
        model,
        X_test,
        stats,
    )

    baseline_loss, baseline_accuracy = calculate_metrics(
        baseline_logits,
        baseline_probs,
        y_test,
    )

    baseline_mdd_probs = baseline_probs[:, 1]

    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"Baseline accuracy: {baseline_accuracy:.6f}")

    summary_rows = []
    epoch_rows = []

    for band_name, (low, high) in BANDS.items():
        print(f"\nRemoving {band_name}: {low}-{high} Hz")

        ablated_logits_list = []
        ablated_probs_list = []

        for start in range(0, len(X_test), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_test))

            raw_batch = X_test[start:end]

            band_component = extract_band(
                raw_batch,
                low,
                high,
            )

            ablated_signal = raw_batch - band_component

            logits, probabilities = evaluate_model(
                model,
                ablated_signal,
                stats,
            )

            ablated_logits_list.append(logits)
            ablated_probs_list.append(probabilities)

            print(f"  Processed {end}/{len(X_test)}")

        ablated_logits = np.concatenate(ablated_logits_list)
        ablated_probs = np.concatenate(ablated_probs_list)

        ablated_mdd_probs = ablated_probs[:, 1]

        ablated_loss, ablated_accuracy = calculate_metrics(
            ablated_logits,
            ablated_probs,
            y_test,
        )

        probability_change = ablated_mdd_probs - baseline_mdd_probs

        loss_change = ablated_loss - baseline_loss
        accuracy_change = ablated_accuracy - baseline_accuracy

        mean_abs_change = np.mean(np.abs(probability_change))
        mean_signed_change = np.mean(probability_change)

        summary_rows.append({
            "fold": fold,
            "band": band_name,
            "low_hz": low,
            "high_hz": high,
            "baseline_loss": baseline_loss,
            "ablated_loss": ablated_loss,
            "loss_increase": loss_change,
            "baseline_accuracy": baseline_accuracy,
            "ablated_accuracy": ablated_accuracy,
            "accuracy_change": accuracy_change,
            "mean_abs_mdd_probability_change": mean_abs_change,
            "mean_signed_mdd_probability_change": mean_signed_change,
        })

        for i in range(len(epoch_indices)):
            epoch_rows.append({
                "fold": fold,
                "band": band_name,
                "epoch_idx": int(epoch_indices[i]),
                "subject_id": int(test_subject_ids[i]),
                "y_true": int(y_test[i]),
                "baseline_p_mdd": float(baseline_mdd_probs[i]),
                "ablated_p_mdd": float(ablated_mdd_probs[i]),
                "probability_change": float(probability_change[i]),
            })

        print(f"  Ablated loss: {ablated_loss:.6f}")
        print(f"  Loss increase: {loss_change:+.6f}")
        print(f"  Accuracy change: {accuracy_change:+.6f}")
        print(f"  Mean absolute probability change: {mean_abs_change:.6f}")

    return summary_rows, epoch_rows


def aggregate_results(all_results):
    df = pd.DataFrame(all_results)

    summary = (
        df.groupby(["band", "low_hz", "high_hz"])
        .agg(
            folds=("fold", "nunique"),
            mean_loss_increase=("loss_increase", "mean"),
            std_loss_increase=("loss_increase", "std"),
            mean_accuracy_change=("accuracy_change", "mean"),
            std_accuracy_change=("accuracy_change", "std"),
            mean_abs_probability_change=(
                "mean_abs_mdd_probability_change", "mean"
            ),
            std_abs_probability_change=(
                "mean_abs_mdd_probability_change", "std"
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_loss_increase",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "importance_rank",
        np.arange(1, len(summary) + 1),
    )

    return df, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MODMA dataset...")
    X, y, subject_ids = load_data()

    print("Dataset shape:", X.shape)

    all_summary_rows = []
    all_epoch_rows = []

    for fold in range(N_FOLDS):
        summary_rows, epoch_rows = run_fold(
            fold,
            X,
            y,
            subject_ids,
        )

        all_summary_rows.extend(summary_rows)
        all_epoch_rows.extend(epoch_rows)

    fold_results, aggregate_summary = aggregate_results(
        all_summary_rows
    )

    fold_results.to_csv(
        OUTPUT_DIR / "band_ablation_all_folds.csv",
        index=False,
    )

    aggregate_summary.to_csv(
        OUTPUT_DIR / "band_ablation_summary.csv",
        index=False,
    )

    pd.DataFrame(all_epoch_rows).to_csv(
        OUTPUT_DIR / "band_ablation_per_epoch.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL-FOLD FREQUENCY BAND ABLATION SUMMARY")
    print("=" * 70)

    print(aggregate_summary.to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_DIR / "band_ablation_all_folds.csv")
    print(OUTPUT_DIR / "band_ablation_summary.csv")
    print(OUTPUT_DIR / "band_ablation_per_epoch.csv")


if __name__ == "__main__":
    main()
