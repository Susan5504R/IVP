import numpy as np
import pandas as pd
import pytest
from src.eval.metrics import calculate_metrics, aggregate_participant_predictions

def test_known_metrics():
    m=calculate_metrics([0,0,1,1],[0,1,1,1],[.1,.7,.8,.9])
    assert m['accuracy']==pytest.approx(.75); assert m['sensitivity']==pytest.approx(1); assert m['specificity']==pytest.approx(.5); assert m['precision']==pytest.approx(2/3); assert m['f1']==pytest.approx(.8); assert m['auc']==pytest.approx(1)
def test_single_class_auc_nan(): assert np.isnan(calculate_metrics([1,1],[1,1],[.8,.9])['auc'])
def test_invalid_shapes_and_scores():
    with pytest.raises(ValueError): calculate_metrics([0,1],[0],[.1,.2])
    with pytest.raises(ValueError): calculate_metrics([0,1],[0,1],[-.1,.2])
def test_participant_aggregation():
    d=pd.DataFrame({'subject_id':[1,1,2,2],'y_true':[1,1,0,0],'p_mdd':[.4,.8,.1,.2]}); p=aggregate_participant_predictions(d)
    assert p.y_pred.tolist()==[1,0]; assert p.n_epochs.tolist()==[2,2]

def test_participant_aggregation_uses_mean_probability_not_majority_vote():
    d=pd.DataFrame({'subject_id':[1,1,1],'y_true':[1,1,1],'p_mdd':[.1,.51,.51]})
    p=aggregate_participant_predictions(d)
    assert p.p_mdd.tolist()==pytest.approx([(.1+.51+.51)/3])
    assert p.y_pred.tolist()==[0]
