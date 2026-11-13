"""Validated binary-classification metrics for MODMA EEG experiments.

This accepts both probability-style scores (0..1) and continuous decision scores
used by SVM-style classifiers. For probability-like scores, the standard 0.5
threshold is used; for decision scores, the default zero threshold is applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _is_probability_like(y_score):
    ys = np.asarray(y_score, dtype=float).reshape(-1)
    return ys.size > 0 and np.isfinite(ys).all() and ((ys >= 0) & (ys <= 1)).all()


def _validate(y_true, y_pred, y_score=None, *, allow_nonprobability=False):
    yt, yp = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if not yt.size or yt.shape != yp.shape:
        raise ValueError(f"Expected non-empty arrays of equal shape; got {yt.shape} and {yp.shape}")
    if not np.isin(yt, [0, 1]).all() or not np.isin(yp, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary: 0=HC, 1=MDD")

    ys = None if y_score is None else np.asarray(y_score, dtype=float).reshape(-1)
    if ys is not None:
        if ys.shape != yt.shape or not np.isfinite(ys).all():
            raise ValueError("y_score must match labels and contain finite values")
        if not allow_nonprobability and not _is_probability_like(ys):
            raise ValueError("y_score must be finite probabilities in [0, 1]")
    return yt.astype(int), yp.astype(int), ys


def calculate_metrics(y_true, y_pred, y_score=None, *, allow_nonprobability=False):
    """Sensitivity=recall(MDD); specificity=recall(HC). AUC is NaN if undefined."""
    yt, yp, ys = _validate(y_true, y_pred, y_score, allow_nonprobability=allow_nonprobability)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(yt, ys)) if ys is not None and np.unique(yt).size == 2 else float("nan")
    sensitivity = float(recall_score(yt, yp, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    precision = float(precision_score(yt, yp, pos_label=1, zero_division=0))
    f1 = float(f1_score(yt, yp, pos_label=1, zero_division=0))
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": sensitivity,
        "f1": f1,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_participant_predictions(epoch_predictions, threshold=None):
    """Mean epoch score per subject, then threshold at 0.5 for probabilities or 0.0 for decision scores."""
    required = {"subject_id", "y_true", "p_mdd"}
    missing = required - set(epoch_predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if threshold is None:
        scores = np.asarray(epoch_predictions["p_mdd"], dtype=float)
        threshold = 0.5 if _is_probability_like(scores) else 0.0

    rows = []
    for sid, group in epoch_predictions.groupby("subject_id", sort=True):
        labels = group.y_true.dropna().unique()
        if len(labels) != 1:
            raise ValueError(f"Subject {sid} has inconsistent ground-truth labels: {labels}")
        score = float(group.p_mdd.mean())
        rows.append({
            "subject_id": sid,
            "y_true": int(labels[0]),
            "p_mdd": score,
            "n_epochs": len(group),
            "y_pred": int(score >= threshold),
        })
    return pd.DataFrame(rows)
