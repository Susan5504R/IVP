
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT / "outputs" / "interpretability"
    / "cdl_mddnet_kfold10_baseline" / "fold_0"
)

df = pd.read_csv(OUTPUT_DIR / "band_importance.csv")

# Plot loss increase.
plt.figure(figsize=(9, 5))
plt.bar(df["band"], df["loss_increase"])
plt.axhline(0, linewidth=1)
plt.xlabel("EEG Frequency Band")
plt.ylabel("Change in Cross-Entropy Loss")
plt.title("Frequency Band Ablation - Loss Change")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "band_loss_change.png", dpi=300)
plt.close()

# Plot mean absolute probability change.
plt.figure(figsize=(9, 5))
plt.bar(
    df["band"],
    df["mean_abs_mdd_probability_change"],
)
plt.xlabel("EEG Frequency Band")
plt.ylabel("Mean Absolute Change in MDD Probability")
plt.title("Frequency Band Ablation - Prediction Sensitivity")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "band_probability_change.png", dpi=300)
plt.close()

print("Saved band ablation plots.")
