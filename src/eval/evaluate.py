"""Evaluate saved fold predictions (no training) and create CSV/ROC/confusion plots.

Example: python -m src.eval.evaluate --runs-dir extracted_archive --out runs/results.csv
"""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc as sk_auc
from src.eval.metrics import calculate_metrics, aggregate_participant_predictions

METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc")
REQUIRED = {"epoch_idx", "subject_id", "y_true", "p_mdd"}


def read_fold_predictions(exp_dir: Path):
    frames = []
    for path in sorted(exp_dir.glob("fold_*/preds.csv"), key=lambda p: int(p.parent.name.split("_")[-1])):
        if not re.fullmatch(r"fold_\d+", path.parent.name):
            continue
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing: raise ValueError(f"{path}: missing {sorted(missing)}")
        if df.empty: warnings.warn(f"Skipping empty file {path}"); continue
        df = df.copy()
        df["p_mdd"] = pd.to_numeric(df.p_mdd, errors="raise")
        if not np.isfinite(df.p_mdd).all() or not df.p_mdd.between(0, 1).all():
            raise ValueError(f"{path}: p_mdd values must be finite probabilities in [0,1]")
        if not df.y_true.isin([0, 1]).all(): raise ValueError(f"{path}: y_true must be 0/1")
        frames.append(df)
    if not frames: raise FileNotFoundError(f"No fold_*/preds.csv under {exp_dir}")
    return frames


def mean_sd(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))


def plot_participant(experiment, pooled, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    part = aggregate_participant_predictions(pooled)
    y, score, pred = part.y_true.to_numpy(), part.p_mdd.to_numpy(), part.y_pred.to_numpy()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, score); area = sk_auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC = {area:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"Participant ROC — {experiment}", xlim=(0,1), ylim=(0,1.02)); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(figures_dir / f"{slug}_participant_roc.png", dpi=180); plt.close(fig)
    else: warnings.warn(f"ROC undefined for {experiment}: only one class present")
    fig, ax = plt.subplots(figsize=(5.5, 5)); ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0,1], display_labels=["HC", "MDD"], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Participant confusion matrix — {experiment}"); fig.tight_layout(); fig.savefig(figures_dir / f"{slug}_participant_confusion_matrix.png", dpi=180); plt.close(fig)


def evaluate_experiment(exp_dir: Path, figures_dir: Path):
    fold_frames = read_fold_predictions(exp_dir); pooled = pd.concat(fold_frames, ignore_index=True)
    cfg_path = exp_dir / "run_config.json"
    protocol = "loso" if "loso" in exp_dir.name.lower() else "kfold10"
    if cfg_path.exists(): protocol = json.loads(cfg_path.read_text()).get("config", {}).get("protocol", protocol)
    if protocol not in {"loso", "kfold10"}: raise ValueError(f"Unsupported protocol {protocol}")
    plot_participant(exp_dir.name, pooled, figures_dir)
    rows = []
    if protocol == "kfold10":
        by_level = {"epoch": [], "participant": []}
        for frame in fold_frames:
            by_level["epoch"].append((frame.y_true, (frame.p_mdd >= .5).astype(int), frame.p_mdd))
            part = aggregate_participant_predictions(frame)
            by_level["participant"].append((part.y_true, part.y_pred, part.p_mdd))
        for level, samples in by_level.items():
            vals = {m: [] for m in METRICS}
            for yt, yp, ys in samples:
                score = calculate_metrics(yt, yp, ys)
                for m in METRICS: vals[m].append(score[m])
            for m in METRICS:
                mean, sd = mean_sd(vals[m]); rows.append(dict(experiment=exp_dir.name, protocol=protocol, level=level, metric=m, mean=mean, sd=sd))
    else:
        part = aggregate_participant_predictions(pooled)
        scores = {"epoch": calculate_metrics(pooled.y_true, (pooled.p_mdd >= .5).astype(int), pooled.p_mdd),
                  "participant": calculate_metrics(part.y_true, part.y_pred, part.p_mdd)}
        for level, vals in scores.items():
            for m in METRICS: rows.append(dict(experiment=exp_dir.name, protocol=protocol, level=level, metric=m, mean=vals[m], sd=float("nan")))
    return rows


def evaluate_runs(runs_dir: Path, output_csv: Path, figures_dir: Path):
    exp_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and list(p.glob("fold_*/preds.csv")))
    if not exp_dirs and list(runs_dir.glob("fold_*/preds.csv")): exp_dirs = [runs_dir]
    if not exp_dirs: raise FileNotFoundError(f"No experiment folders with fold predictions under {runs_dir}")
    rows = []
    for exp in exp_dirs:
        exp_rows = evaluate_experiment(exp, figures_dir); rows.extend(exp_rows)
        print(f"{exp.name}: evaluated {len(list(exp.glob('fold_*/preds.csv')))} fold prediction files")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["experiment", "protocol", "level", "metric", "mean", "sd"])
    df.to_csv(output_csv, index=False, float_format="%.8f")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--runs-dir", type=Path, default=Path("runs")); p.add_argument("--out", type=Path, default=Path("runs/results.csv")); p.add_argument("--figures", type=Path, default=Path("reports/figures")); a=p.parse_args()
    df=evaluate_runs(a.runs_dir,a.out,a.figures); print(f"Wrote {len(df)} rows to {a.out}; plots in {a.figures}")
if __name__ == "__main__": main()
