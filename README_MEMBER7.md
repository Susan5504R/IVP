# Member 7 — Unified evaluation, plots and CLI demo

## Files added

- `src/eval/metrics.py`: validated binary metrics and participant aggregation.
- `src/eval/evaluate.py`: discovers saved fold predictions; creates the required unified CSV and plots without training.
- `src/app/predict.py`: one-epoch inference using a matching checkpoint and fold normalizer.
- `tests/test_predict.py`, `tests/test_metrics.py`, `tests/test_evaluate.py`: inference, metric, aggregation and evaluation tests.
- `runs/results.csv`: required 72-row unified metrics file.
- `runs/*_reconstructed.csv`: combined epoch and participant predictions reconstructed from fold CSVs.
- `reports/figures/`: six participant ROC curves and six confusion matrices.
- `reports/pooled_participant_audit.csv`: pooled participant audit, including confusion matrix counts.
- `reports/member7_evaluation_audit.md`: methods, run counts, caveats and reproducibility notes.
- `reports/member7_metric_validation.csv`: original and independently recomputed values for all 72 metric rows, with differences and pass/fail status.
- `reports/member7_prediction_integrity.csv`: fold-to-combined and fold-to-participant prediction reconciliation for all six experiments.

## Evaluate archived experiments (no training)

Download `final_CDL_MDDNet_all_experiments.zip` from the [v1.0.0 GitHub release](https://github.com/Susan5504R/IVP/releases/tag/v1.0.0). The handover records SHA-256 `cfe747a9b864554df9bd34066bb3c2bdea3f4a012cc2d1e2e196117bdede2146`. Verify it before extracting:

```powershell
$archive = "final_CDL_MDDNet_all_experiments.zip"
Get-FileHash $archive -Algorithm SHA256
Expand-Archive $archive -DestinationPath "runs\released"
```

Install with Python 3.10+ from the repository root (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest -q tests/test_predict.py tests/test_metrics.py tests/test_evaluate.py
python -m pytest tests/ -v
```

The archive contains six completed experiment directories, saved fold predictions, inference checkpoints, and fold normalizers. It does not contain raw EEG or the processed EEG dataset. Do not retrain to reproduce the evaluation or demo.

The evaluator reads `experiment/fold_N/preds.csv` and identifies `kfold10` vs `loso` from `run_config.json`. The results schema is exactly `experiment, protocol, level, metric, mean, sd`; values are fractions, not percentages. K-fold reports mean and sample SD across folds. LOSO pools the out-of-fold predictions and leaves SD blank because every subject is held out once. Reproduce into new paths so accepted output files are not overwritten:

```powershell
python -m src.eval.evaluate --runs-dir runs/released --out reports/member7_reproduced_results.csv --figures reports/member7_reproduced_figures
```

## Participant-level convention

Participant scores reproduce the archived rule: mean epoch `p_mdd` per participant, then threshold at 0.5. This is **not** majority voting over thresholded epoch labels. Epoch-level classification is `p_mdd >= 0.5` for MDD (`1`); otherwise HC (`0`). ROC and confusion-matrix plots use pooled participant-level predictions.

## CLI demo

The CLI accepts one already preprocessed, finite real-valued NumPy epoch with shape `(channels, 1000)`: one four-second window at 250 Hz. It does not load raw EEG, filter, run ICA, select channels, or create epochs. Use the same channel order and preprocessing as training: E1-E128 for the standard montage, 50 Hz notch (Q=30), fourth-order 0.5-45 Hz zero-phase Butterworth filter, the documented 32-component FastICA cleaning (seed 42, remove up to three components with absolute kurtosis above 5), and non-overlapping four-second epoching. Reduced-montage models need the corresponding training channel selection and fold artifacts; do not guess missing preprocessing rules.

Normalization is not refit. Each fold's `normalizer_stats.npz` contains training-only per-channel mean/std, epsilon, and channel count from that fold's inner training participants. Always pair it with the `model.pt` from the same experiment and fold. The CLI loads the inference-only state dict strictly; it does not accept a training-resume checkpoint.

Verify both files exist before running; incomplete folds must not be used:

```powershell
$fold = "runs\released\cdl_mddnet_kfold10_baseline\fold_0"
Test-Path "$fold\model.pt"
Test-Path "$fold\normalizer_stats.npz"
python -m src.app.predict --epoch one_epoch.npy --checkpoint "$fold\model.pt" --normalizer "$fold\normalizer_stats.npz"
```

Expected output format:

```text
Prediction: HC | P(MDD): 0.1234 | threshold: 0.5
Research prototype only; not for clinical use.
```

The script uses `src.models.cdl_mddnet.CDLMDDNet`. It is a research prototype, not a clinically validated diagnostic system. The experiments use a single-site cohort of 53 participants and do not establish clinical utility or generalization. A single-epoch output is not a participant diagnosis; the CLI does not aggregate epochs to a participant score.

## Known artifact limitation

The LOSO DANN archive includes `fold_47/preds.csv` and its normalizer/training artifacts but lacks `fold_47/model.pt`. Its saved predictions remain usable for evaluation, but it must not be used for inference. No experiment was retrained. This repository working tree does not include raw EEG or processed epochs.
