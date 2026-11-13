# CDL-MDDNet: Lightweight Subject-Independent EEG Classification for Major Depressive Disorder

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-93%20Passing-brightgreen.svg)]()

CDL-MDDNet is a lightweight, interpretable convolutional neural network architecture engineered for subject-independent detection of Major Depressive Disorder (MDD) from high-density 128-channel resting-state EEG recordings. Evaluated on the Multi-modal Open Dataset for Mental-disorder Analysis (MODMA) across 53 participants, this project enforces strict zero-leakage cross-validation and features a comprehensive channel-reduction study to optimize for hardware efficiency.

*Image and Video Processing course project, 7-person team.*

---

## 1. Team & Responsibilities

| Path / Component | Responsibilities | Owner |
| :--- | :--- | :--- |
| `src/models/cdl_mddnet.py` | CDL-MDDNet network architecture, domain-generalization (DANN) | Member 1 |
| `src/data/` | MODMA loading, filtering, ICA, epoching, and signal normalizer | Member 2 |
| `src/training/` | CV Splits, training loop, and canonical experiment runner | Member 3 |
| `src/eval/` (Partial) | Benchmark testing, leakage tests, CI, results table, slide automation | Member 4 |
| `src/explain/` | Captum attribution, topomaps, and channel importance ranking | Member 5 |
| `src/models/ms_mddnet.py` | **MS-MDDNet baseline validation, hardware-efficiency Channel-Reduction** | **Member 6** |
| `src/eval/` (Partial) | Unified evaluation, metrics validation, plots, and CLI demo | Member 7 |

---

## 2. Setup & Installation

Python 3.10+ is required. Training runs best on Google Colab (T4 GPU). 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the environment and check for data leakage via the test suite:
```bash
pytest -q tests/test_predict.py tests/test_metrics.py tests/test_evaluate.py
python -m pytest tests/ -v
```
*(Tests enforce zero subject leakage across 10-fold and LOSO splits).*

---

## 3. Dataset & Contracts

The project requires the **MODMA** dataset (53 participants: 24 MDD, 29 Healthy Controls). Raw EEG is NOT included in the repository.

1. Download from `modma.lzu.edu.cn`.
2. Processed epochs must be placed at `data/processed/modma_epochs.npz`.
3. **Format Contract:** `X` float32 `(N, C, 1000)`, `y` (0 healthy, 1 MDD), `subject_id` int, `ch_names` list.
4. **Normalizer Contract (`src/data/normalize.py`):** `stats = fit_normalizer(train_x)`, `out = apply_normalizer(x, stats)`. Fit on the training fold only to prevent leakage.
5. **Results Contract:** `runs/results.csv` with schema: `experiment, protocol, level, metric, mean, sd`. Values are fractions (e.g., 0.87).

---

## 4. Training Experiments (Deep Learning)

All core models are trained using the central runner (`src.training.run_experiment`). Artifacts (predictions, checkpoints, fold normalizers) are saved to `runs/`.

**Foundational Baselines & DANN (Experiments 1-6)**
```bash
# 10-Fold Subject-Wise Baseline (ERM)
python -m src.training.run_experiment --config config/train_config.json --data data/processed/modma_epochs.npz --experiment cdl_mddnet_kfold10_baseline --protocol kfold10 --no-dann

# 10-Fold Subject-Wise DANN
python -m src.training.run_experiment --config config/train_config.json --data data/processed/modma_epochs.npz --experiment cdl_mddnet_kfold10_dann --protocol kfold10 --use-dann --dann-gamma 10.0

# 53-Fold Leave-One-Subject-Out (LOSO) Baseline
python -m src.training.run_experiment --config config/train_config.json --data data/processed/modma_epochs.npz --experiment cdl_mddnet_loso_baseline --protocol loso --no-dann
```

---

## 5. Channel Reduction Study

To optimize for hardware efficiency (<100k parameters and minimal sensor footprint), the CDL-MDDNet was evaluated on progressively reduced spatial montages. Channels were selected using the attribution rankings provided by Member 5.

**Execution:**
```bash
# Target: 32 Channels
python -m src.training.run_experiment \
  --config config/train_config.json \
  --data data/processed/modma_epochs.npz \
  --experiment channel_study_32 \
  --protocol kfold10 \
  --channel-ranking outputs/interpretability/cdl_mddnet_kfold10_baseline/channel_ranking.csv \
  --top-k-channels 32

# Standard 10-20 System Baseline (16 Channels)
python -m src.training.run_experiment \
  --config config/train_config.json \
  --data data/processed/modma_epochs.npz \
  --experiment channel_study_16 \
  --protocol kfold10 \
  --channels E22 E9 E24 E124 E36 E104 E52 E92 E70 E83 E33 E122 E47 E108 E58 E96
```

---

## 6. Classical ML Baseline (FBCSP + RBF-SVM)

For baseline comparison, we evaluated a Filter Bank Common Spatial Pattern (FBCSP) feature extractor paired with an RBF-kernel SVM (see `MODMA_FBCSP_SVM_10Fold_and_53_LOSO_Tuned.ipynb`). 

*   **Filter bank:** Zero-phase Butterworth band-pass (order 4) into five bands (delta, theta, alpha, beta, gamma).
*   **CSP:** 4 filters per band, 20 features total. Fitted strictly on training participants.
*   **Classifier:** `StandardScaler` + `SVC(kernel='rbf', class_weight='balanced')`.
*   **Results (Pooled Participant-Level):**
    *   *10-fold CV (Baseline):* 64.15% Accuracy, 64.80% ROC-AUC.
    *   *53-fold LOSO (Tuned):* 66.04% Accuracy, 63.22% ROC-AUC.

---

## 7. Unified Evaluation & CLI Demo

### Evaluate Archived Experiments (No Training)
The evaluator reads `preds.csv` files generated across folds and outputs the required 72-row unified metrics CSV alongside aggregated ROC and Confusion Matrix plots.

```bash
python -m src.eval.evaluate \
  --runs-dir runs/released \
  --out reports/member7_reproduced_results.csv \
  --figures reports/member7_reproduced_figures
```

### CLI Inference Demo
The CLI accepts a single preprocessed 4-second NumPy epoch `(channels, 1000)`. It requires an exact match between the inference checkpoint and the fold's normalization statistics.

```bash
$fold = "runs\released\cdl_mddnet_kfold10_baseline\fold_0"
python -m src.app.predict \
  --epoch one_epoch.npy \
  --checkpoint "$fold\model.pt" \
  --normalizer "$fold\normalizer_stats.npz"
```
*Note: This is a research prototype. Single-epoch output does not constitute a clinical participant diagnosis.*

---

## 8. Summary of Verified Results

### Foundational Accuracies (128-Channel)
Participant-level metrics evaluated on all 53 unseen MODMA participants:
*   **MS-MDDNet Baseline (10-Fold ERM):** 62.26% Participant Accuracy | 53.00% Pooled Epoch Accuracy
*   **CDL-MDDNet (10-Fold DANN):** 47.17% Participant Accuracy | 46.96% Pooled Epoch Accuracy 

### Channel Reduction Target Results
The primary objective was to reduce the EEG montage to 32 channels or fewer without sacrificing more than 3 percentage points of accuracy relative to the 128-channel full montage. 

| Channels | Parameters | Pooled Accuracy (%) | AUC (%) |
| :--- | :--- | :---: | :---: |
| 128 (Full Montage) | 25,627 | 46.96 | 42.59 |
| 64 (Top-K) | 24,603 | 46.37 | 43.50 |
| **32 (Project Target)** | **24,091** | **45.24** | **44.36** |
| 16 (10-20 System) | 23,835 | 39.22 | 39.83 |
| 8 (Extreme Reduction) | 23,707 | 46.46 | 46.53 |

**Conclusion:** Reducing the montage to the targeted 32 channels resulted in a minimal accuracy degradation of only **1.72 points**, successfully satisfying the project's threshold mandate.