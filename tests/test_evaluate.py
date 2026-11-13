import json
import pandas as pd
from src.eval.evaluate import evaluate_experiment

def test_evaluate_one_fold_experiment(tmp_path):
    exp = tmp_path / 'toy_kfold'
    fold = exp / 'fold_0'
    fold.mkdir(parents=True)
    (exp / 'run_config.json').write_text(json.dumps({'config': {'protocol': 'kfold10'}}))
    pd.DataFrame({
        'epoch_idx': [0, 1, 2, 3],
        'subject_id': [1, 1, 2, 2],
        'y_true': [1, 1, 0, 0],
        'p_mdd': [.8, .7, .1, .2],
    }).to_csv(fold / 'preds.csv', index=False)
    rows = evaluate_experiment(exp, tmp_path / 'figures')
    assert len(rows) == 16
    assert {'balanced_accuracy', 'recall'} <= {row['metric'] for row in rows}
    assert (tmp_path / 'figures' / 'toy_kfold_participant_roc.png').exists()
    assert (tmp_path / 'figures' / 'toy_kfold_participant_confusion_matrix.png').exists()
