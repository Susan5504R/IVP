
"""Detailed performance diagnostics across K-fold test predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "cdl_mddnet_kfold10_baseline"
RUN_DIR = ROOT / "runs" / EXPERIMENT
OUTPUT_DIR = ROOT / "outputs" / "interpretability" / EXPERIMENT


def main():
    rows = []

    for fold in range(10):
        fold_dir = RUN_DIR / f"fold_{fold}"
        preds = pd.read_csv(fold_dir / "preds.csv")

        y_true = preds["y_true"].to_numpy(dtype=int)
        p_mdd = preds["p_mdd"].to_numpy(dtype=float)
        y_pred = (p_mdd >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        auc = roc_auc_score(y_true, p_mdd)

        rows.append({
            "fold": fold,
            "n_test": len(y_true),
            "hc_count": int(np.sum(y_true == 0)),
            "mdd_count": int(np.sum(y_true == 1)),
            "predicted_hc": int(np.sum(y_pred == 0)),
            "predicted_mdd": int(np.sum(y_pred == 1)),
            "mean_p_mdd": float(np.mean(p_mdd)),
            "std_p_mdd": float(np.std(p_mdd)),
            "min_p_mdd": float(np.min(p_mdd)),
            "max_p_mdd": float(np.max(p_mdd)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "auc": auc,
        })

        print(f"\n========== FOLD {fold} ==========")
        print("Confusion matrix:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["HC", "MDD"],
            zero_division=0,
        ))

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR / "detailed_prediction_diagnostics.csv",
        index=False,
    )

    print("\n========== SUMMARY ==========")
    print(result.to_string(index=False))

    print("\nSaved detailed_prediction_diagnostics.csv")


if __name__ == "__main__":
    main()
