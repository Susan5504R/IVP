
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT / "outputs" / "interpretability"
    / "cdl_mddnet_kfold10_baseline" / "fold_0"
)

IG_PATH = OUTPUT_DIR / "integrated_gradients.npz"

data = np.load(IG_PATH, allow_pickle=False)

attributions = data["attributions"]
y_true = data["y_true"]
ch_names = data["ch_names"].astype(str)
delta = data["convergence_delta"]

assert attributions.shape == (450, 128, 1000)

# Absolute attribution values.
abs_attr = np.abs(attributions)

# Split by actual ground-truth labels.
hc_attr = abs_attr[y_true == 0]
mdd_attr = abs_attr[y_true == 1]

# Average over epochs and time points.
hc_scores = hc_attr.mean(axis=(0, 2))
mdd_scores = mdd_attr.mean(axis=(0, 2))

hc_ranking = pd.DataFrame({
    "channel": ch_names,
    "score": hc_scores,
}).sort_values("score", ascending=False)

mdd_ranking = pd.DataFrame({
    "channel": ch_names,
    "score": mdd_scores,
}).sort_values("score", ascending=False)

hc_ranking.to_csv(OUTPUT_DIR / "hc_channel_ranking.csv", index=False)
mdd_ranking.to_csv(OUTPUT_DIR / "mdd_channel_ranking.csv", index=False)

# Compare rankings.
comparison = pd.DataFrame({
    "channel": ch_names,
    "hc_score": hc_scores,
    "mdd_score": mdd_scores,
})

comparison["score_difference"] = (
    comparison["mdd_score"] - comparison["hc_score"]
)

comparison["absolute_difference"] = (
    comparison["score_difference"].abs()
)

comparison.to_csv(
    OUTPUT_DIR / "hc_mdd_channel_comparison.csv",
    index=False,
)

# Spearman rank correlation.
spearman = hc_ranking.set_index("channel")["score"].corr(
    mdd_ranking.set_index("channel")["score"],
    method="spearman",
)

# Top-15 overlap.
hc_top15 = set(hc_ranking.head(15)["channel"])
mdd_top15 = set(mdd_ranking.head(15)["channel"])

overlap = hc_top15.intersection(mdd_top15)

print("HC epochs:", len(hc_attr))
print("MDD epochs:", len(mdd_attr))

print("\nTop 15 HC channels:")
print(hc_ranking.head(15).to_string(index=False))

print("\nTop 15 MDD channels:")
print(mdd_ranking.head(15).to_string(index=False))

print("\nSpearman rank correlation:", spearman)
print("Top-15 overlap:", len(overlap), "/ 15")
print("Shared channels:", sorted(overlap))

print("\nConvergence delta:")
print("Mean absolute:", np.mean(np.abs(delta)))
print("Median absolute:", np.median(np.abs(delta)))
print("Maximum absolute:", np.max(np.abs(delta)))

print("\nSaved group-wise rankings and comparison.")
