# Member 7 evaluation audit — 10 November 2026

## Execution

- Input archive: `final_CDL_MDDNet_all_experiments (1).zip` (897 entries).
- Discovered and evaluated: 6 experiments.
- K-fold experiments: 4 × 10 folds.
- LOSO experiments: 2 × 53 folds.
- Fold prediction files successfully read: 146.
- Epoch predictions across each full experiment: 4,002.
- Participants in each pooled evaluation: 53.
- Unified `runs/results.csv`: 72 rows (6 experiments × 2 levels × 6 required metrics).
- Participant-level figures: 6 ROC curves and 6 confusion matrices.
- Unit/integration tests in the original audit run: 5 passed; see Current Verification for the refreshed full suite.

## Evaluation convention

Epoch predictions use `p_mdd >= 0.5`. Participant scores are the arithmetic mean of each participant's epoch probabilities, then thresholded at 0.5. This matches the supplied participant prediction CSV convention. It is not equivalent to majority vote over thresholded epoch labels; the handover/README phrase "majority vote" should be clarified with the team before altering this behavior.

For `kfold10`, the unified CSV reports mean and sample standard deviation across folds. For LOSO, each participant is held out exactly once, so metrics are computed on pooled out-of-fold predictions and SD is blank. ROC and confusion-matrix plots use pooled participant-level predictions in all protocols.

## Integrity and missing artifacts

The LOSO DANN archive contains all 53 fold prediction CSVs, including `fold_47/preds.csv`, but does not contain `fold_47/model.pt`. Its predictions can therefore be included in evaluation, while the missing checkpoint cannot be used for inference. No model training was rerun.

The two k-fold experiment folders without root-level combined/participant aggregate CSVs were reconstructed from their fold prediction CSVs. Reconstructed files are saved under `runs/` with `_reconstructed` suffixes.

## Pooled participant-level results

See `reports/pooled_participant_audit.csv` for accuracy, balanced accuracy, ROC-AUC, and confusion-matrix counts. These pooled values are useful for the overall audit and figures; they are distinct from the fold mean ± SD in `runs/results.csv` for k-fold experiments.

## Verification

```bash
pytest -q
python -m src.eval.evaluate --runs-dir archive --out reports/member7_reproduced_results.csv --figures reports/member7_reproduced_figures
```

No raw EEG dataset, model checkpoints, or training archives are part of the deliverable source package.

## Release Artifact Verification

The v1.0.0 release archive was downloaded for the inference audit and its SHA-256 matched the handover value `cfe747a9b864554df9bd34066bb3c2bdea3f4a012cc2d1e2e196117bdede2146`. It contains 146 fold directories across six experiments: 40 k-fold and 106 LOSO folds. A paired `model.pt` and `normalizer_stats.npz` exists for 145 folds. The only missing inference checkpoint is `cdl_mddnet_loso_dann/fold_47/model.pt`; this fold remains available to saved-prediction evaluation but is not eligible for inference.

The demo's documented example, `cdl_mddnet_kfold10_baseline/fold_0`, contains both files (`model.pt`: 110,170 bytes; `normalizer_stats.npz`: 2,026 bytes). Its run configuration identifies the full 128-channel, 1,000-sample montage (`E1`-`E128`) and ERM model. The two artifacts are from the same fold.

The release `final_verified_results.csv` and repository copy have identical CSV content after line-ending normalization (the repository copy uses CRLF). The released baseline `combined_preds.csv` is byte-identical to the corresponding reconstructed prediction file in `runs/`. No accepted metrics or prediction files were edited for this task.

The working tree's `data/raw/` and `data/processed/` directories contain only `.gitkeep`; neither raw recordings nor preprocessed EEG epochs are available locally. Therefore a local real-EEG sample comparison is unavailable. An end-to-end inference smoke test used the verified released checkpoint and matching fold normalizer with a temporary zero-valued synthetic `(128, 1000)` input. The CPU CLI completed with `Prediction: HC | P(MDD): 0.2281 | threshold: 0.5`. This is a technical smoke test only, not a real participant prediction, and must not be presented as one.

## Current Verification

- Predictor tests: `15 passed`, including the CLI missing-input error case.
- Full repository command `python -m pytest tests/ -v`: `114 passed, 1 warning, 0 skipped`; the warning is scikit-learn's expected single-class AUC warning.
- Release archive evaluation: regenerated all six experiments and 72 metric rows to temporary paths. The row keys and all serialized `mean`/`sd` values exactly match `runs/results.csv`.
- Independent recomputation directly from the 146 fold prediction CSVs: `72/72` comparisons pass at tolerance `1e-8`; maximum absolute difference is `4.93e-9` for means and `4.93e-9` for SDs, attributable to the accepted table's eight-decimal storage precision. See `reports/member7_metric_validation.csv` for each original value, recomputed value, difference, and status.
- Independent prediction aggregation: all six reconstructed combined and participant-level CSVs reconcile with fold predictions; archived aggregates reconcile wherever supplied. There are 4,002 unique epoch rows and 53 participants per experiment, with no duplicate participant/epoch pairs or inconsistent labels. See `reports/member7_prediction_integrity.csv`.
- All 12 accepted participant plots (six ROC and six confusion matrices) were visually inspected. Titles, class/axis labels, chance legend, AUC labels, and confusion counts match the pooled participant predictions.
- The release `final_verified_results.csv` matches the repository version after line-ending normalization; the baseline combined prediction file is byte-identical to its reconstructed repository copy. Accepted metrics and prediction files were not edited.
- The participant-level convention is verified in `src/eval/metrics.py` and `src/eval/evaluate.py`: mean epoch probability per participant followed by `>= 0.5`, not majority voting over binary epoch labels. A regression test covers a case where those rules produce different predictions.
