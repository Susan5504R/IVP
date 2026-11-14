# CDL-MDDNet: Lightweight Subject-Independent EEG Classification for Major Depressive Disorder

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-93%20Passing-brightgreen.svg)]()

CDL-MDDNet is a lightweight, interpretable convolutional neural network architecture engineered for subject-independent detection of Major Depressive Disorder (MDD) from high-density 128-channel resting-state EEG recordings. Evaluated on the Multi-modal Open Dataset for Mental-disorder Analysis (MODMA) across 53 participants, this project enforces strict zero-leakage cross-validation and features a comprehensive channel-reduction study to optimize for hardware efficiency.

*Image and Video Processing course project, 7-person team.*

---

## Table of Contents
1. [Team & Responsibilities](#1-team--responsibilities)
2. [Environment Setup & Installation](#2-environment-setup--installation)
3. [Dataset Preparation & Contracts](#3-dataset-preparation--contracts)
4. [Training & Deep Learning Experiments](#4-training--deep-learning-experiments)
5. [Classical Machine Learning Baseline (FBCSP + RBF-SVM)](#5-classical-machine-learning-baseline-fbcsp--rbf-svm)
6. [Comparative Literature Baseline (Lower Accuracy 10-Fold Benchmark)](#6-comparative-literature-baseline-lower-accuracy-10-fold-benchmark)
7. [Channel Reduction Study](#7-channel-reduction-study)
8. [Unified Evaluation & CLI Demo](#8-unified-evaluation--cli-demo)
9. [Summary of Verified Results & Visualizations](#9-summary-of-verified-results--visualizations)
10. [Project Repository Layout](#10-project-repository-layout)

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

## 2. Environment Setup & Installation

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

## 3. Dataset Preparation & Contracts

The project requires the **MODMA** dataset (53 participants: 24 MDD, 29 Healthy Controls). Raw EEG is NOT included in the repository.

1. Download from `modma.lzu.edu.cn`.
2. Processed epochs must be placed at `data/processed/modma_epochs.npz`.
3. **Format Contract:** `X` float32 `(N, C, 1000)`, `y` (0 healthy, 1 MDD), `subject_id` int, `ch_names` list.
4. **Normalizer Contract (`src/data/normalize.py`):** `stats = fit_normalizer(train_x)`, `out = apply_normalizer(x, stats)`. Fit on the training fold only to prevent leakage.
5. **Results Contract:** `runs/results.csv` with schema: `experiment, protocol, level, metric, mean, sd`. Values are fractions (e.g., 0.87).

---

## 4. Training & Deep Learning Experiments

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

## 5. Classical Machine Learning Baseline (FBCSP + RBF-SVM)

For robust baseline comparison, we evaluated a Filter Bank Common Spatial Pattern (FBCSP) feature extractor paired with an RBF-kernel SVM (implemented in `MODMA_FBCSP_SVM_10Fold_and_53_LOSO_Tuned.ipynb`).

* **Filter bank:** Zero-phase Butterworth band-pass (order 4) into five bands (delta: 1–4 Hz, theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–30 Hz, gamma: 30–45 Hz).
* **CSP:** 4 filters per band, yielding 20 features total. Fitted strictly on training participants.
* **Classifier:** `StandardScaler` + `SVC(kernel='rbf', class_weight='balanced')`.
* **Pooled Participant-Level Results:**
  * **10-fold CV (Fixed SVM, C=1.0):** **64.15% Accuracy**, 64.37% Balanced Accuracy, **64.80% ROC-AUC** (Confusion Matrix: TN=18, FP=11, FN=8, TP=16).
  * **53-fold LOSO (Tuned SVM):** **66.04% Accuracy**, 66.09% Balanced Accuracy, **63.22% ROC-AUC** (Confusion Matrix: TN=19, FP=10, FN=8, TP=16).

---

## 6. Comparative Literature Baseline (Lower Accuracy 10-Fold Benchmark)

To rigorously contextualize our findings against published literature that reports modest performance under strict subject-independent 10-fold cross-validation, we incorporate the foundational benchmark study by AlAqel et al. (MS-MDDNet, Diagnostics 2026). 

* **Protocol:** Strict 10-fold subject-disjoint cross-validation on the MODMA dataset (preventing participant memorization).
* **Performance Benchmark:** Achieves a pooled epoch accuracy of **53.00%** and a participant-level baseline accuracy of **62.26%** (Balanced Accuracy: 62.64%, ROC-AUC: 0.5632).
* **Significance for Our Work:** This lower-accuracy, leakage-free reference establishes that realistic cross-subject EEG generalisation hovers around 60–66%, directly contrasting with overly optimistic segment-level splits (>90%) found in legacy literature. Our CDL-MDDNet ERM model matches this rigorous baseline at **62.26%** participant accuracy.

---

## 7. Channel Reduction Study

To optimize for hardware efficiency (<100k parameters and minimal sensor footprint), CDL-MDDNet was evaluated on progressively reduced spatial montages using Captum attribution rankings.

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

### Channel Reduction Target Results
| Channels | Parameters | Pooled Accuracy (%) | AUC (%) | Status |
| :--- | :--- | :---: | :---: | :--- |
| 128 (Full Montage) | 25,627 | 46.96 | 42.59 | Baseline |
| 64 (Top-K) | 24,603 | 46.37 | 43.50 | Pass |
| **32 (Project Target)** | **24,091** | **45.24** | **44.36** | **Pass (<3pt Drop Mandate)** |
| 16 (10-20 System) | 23,835 | 39.22 | 39.83 | Steep Drop |
| 8 (Extreme Reduction) | 23,707 | 46.46 | 46.53 | Robust |

*Conclusion:* Reducing the montage to the targeted 32 channels resulted in a minimal accuracy drop of only **1.72 points**, satisfying the project's tolerance mandate.

---

## 8. Unified Evaluation & CLI Demo

### Evaluate Archived Experiments (No Training)
```bash
python -m src.eval.evaluate \
  --runs-dir runs/released \
  --out reports/member7_reproduced_results.csv \
  --figures reports/member7_reproduced_figures
```

### CLI Inference Demo
```bash
$fold = "runs\released\cdl_mddnet_kfold10_baseline\fold_0"
python -m src.app.predict \
  --epoch one_epoch.npy \
  --checkpoint "$fold\model.pt" \
  --normalizer "$fold\normalizer_stats.npz"
```
*Note: Research prototype only; not for clinical use.*

---

## 9. Summary of Verified Results & Visualizations

Participant-level metrics evaluated on all 53 unseen MODMA participants:

| Model / Experiment | Protocol | Accuracy | Balanced Acc | ROC-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FBCSP + RBF-SVM (Tuned)** | 53-Fold LOSO | **66.04%** | **66.09%** | **0.6322** | TN=19, FP=10, FN=8, TP=16 |
| **FBCSP + RBF-SVM (Fixed)** | 10-Fold | **64.15%** | **64.37%** | **0.6480** | TN=18, FP=11, FN=8, TP=16 |
| **MS-MDDNet / CDL-MDDNet Baseline** | 10-Fold ERM | **62.26%** | **62.64%** | **0.5632** | TN=17, FP=12, FN=8, TP=16 |
| **CDL-MDDNet (DANN)** | 10-Fold | **47.17%** | **47.05%** | **0.4770** | TN=14, FP=15, FN=13, TP=11 |
| **CDL-MDDNet (LOSO Baseline)** | 53-Fold LOSO | **37.74%** | **38.07%** | **0.3405** | TN=10, FP=19, FN=14, TP=10 |

---

## 10. Project Repository Layout

```text
IVP/
├── README.md                           # Comprehensive project documentation
├── requirements.txt                    # Project dependencies
├── config/
│   └── train_config.json               # Baseline hyperparameters
├── data/
│   └── processed/modma_epochs.npz      # Preprocessed 128-channel EEG tensors
├── reports/
│   ├── figures/                        # ROC, Confusion Matrices, Channel Reduction Plots
│   └── member7_evaluation_audit.md     # Evaluation methods and reproducibility notes
├── runs/                               # Saved predictions, checkpoints, and normalizers
├── src/
│   ├── app/predict.py                  # CLI single-epoch predictor
│   ├── data/                           # MODMA loading, filtering, ICA, and normalizer
│   ├── eval/                           # Benchmarking, results parsing, and slide updating
│   ├── explain/                        # Captum attribution and channel rankings
│   ├── models/cdl_mddnet.py            # Main Model and GRL architecture
│   └── training/run_experiment.py      # Core experiment CLI runner
└── tests/                              # Pytest suite (Inference, Leakage, Integration)
```