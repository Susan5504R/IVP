"""Validated binary-classification metrics for CDL-MDDNet (0=HC, 1=MDD)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _validate(y_true, y_pred, y_score=None):
    yt, yp = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if not yt.size or yt.shape != yp.shape:
        raise ValueError(f"Expected non-empty arrays of equal shape; got {yt.shape} and {yp.shape}")
    if not np.isin(yt, [0, 1]).all() or not np.isin(yp, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary: 0=HC, 1=MDD")
    ys = None if y_score is None else np.asarray(y_score, dtype=float).reshape(-1)
    if ys is not None and (ys.shape != yt.shape or not np.isfinite(ys).all() or ((ys < 0) | (ys > 1)).any()):
        raise ValueError("y_score must match labels and contain finite probabilities in [0, 1]")
    return yt.astype(int), yp.astype(int), ys


def calculate_metrics(y_true, y_pred, y_score=None):
    """Sensitivity=recall(MDD); specificity=recall(HC). AUC is NaN if undefined."""
    yt, yp, ys = _validate(y_true, y_pred, y_score)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(yt, ys)) if ys is not None and np.unique(yt).size == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "sensitivity": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def aggregate_participant_predictions(epoch_predictions):
    """Reproduce archive rule: mean epoch P(MDD) per subject, then threshold at .5."""
    import pandas as pd
    required = {"subject_id", "y_true", "p_mdd"}
    missing = required - set(epoch_predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = []
    for sid, group in epoch_predictions.groupby("subject_id", sort=True):
        labels = group.y_true.dropna().unique()
        if len(labels) != 1:
            raise ValueError(f"Subject {sid} has inconsistent ground-truth labels: {labels}")
        score = float(group.p_mdd.mean())
        rows.append({"subject_id": sid, "y_true": int(labels[0]), "p_mdd": score,
                     "n_epochs": len(group), "y_pred": int(score >= .5)})
    return pd.DataFrame(rows)
