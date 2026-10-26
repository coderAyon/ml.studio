import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


def fixture():
    rng=np.random.default_rng(23)
    return pd.DataFrame({'signal':rng.normal(size=120),'noise':rng.normal(size=120),
                         'target':rng.integers(0,2,120)})


def test_holdout_records_are_actual_positions_not_index_labels():
    from mlstudio.training import train
    f=fixture();f.index=[1]*len(f)
    r=train(f,'target','Logistic Regression')
    diag=r.metadata['diagnostics'];positions=r.metadata['test_positions']
    _,expected=train_test_split(np.arange(len(f)),test_size=.25,random_state=42,stratify=f.target)
    assert positions==expected.tolist()
    assert diag['total']==30 and diag['shown']==30
    assert {x['position'] for x in diag['records']}==set(positions)
    for row in diag['records']:
        assert row['actual']==f.iloc[row['position']].target
        assert row['correct']==(row['actual']==row['predicted'])
    assert diag['roc']['positive_class']=='1'
    assert diag['baseline']['metric']=='Accuracy'


def test_regression_diagnostics_and_feature_explanation():
    from mlstudio.training import train
    from mlstudio.evaluation import explain_inputs
    f=fixture();f['target']=f.signal*8+np.random.default_rng(2).normal(0,.01,len(f))
    r=train(f,'target','Ridge')
    rows=r.metadata['diagnostics']['records']
    assert rows[0]['absolute_error']>=rows[-1]['absolute_error']
    explanation=explain_inputs(r,f)
    assert explanation['rows'][0]['feature']=='signal'
    changed=f.copy();changed.loc[0,'signal']=100
    with pytest.raises(ValueError,match='same dataset'):explain_inputs(r,changed)


def test_batch_prediction_mapping_probabilities_and_no_mutation():
    from mlstudio.training import train
    from mlstudio.prediction import prediction_table
    f=fixture();r=train(f,'target','Logistic Regression')
    source=f.head().rename(columns={'signal':'renamed'})
    with pytest.raises(ValueError):prediction_table(r,source)
    output,summary=prediction_table(r,source,{'signal':'renamed','noise':'noise'},allow_extra=True)
    assert len(output)==5 and summary['probabilities']
    assert len(summary['probability_columns'])==2
    np.testing.assert_allclose(output[summary['probability_columns']].sum(axis=1),1)
    assert 'prediction' not in source


def test_model_only_prediction_export_without_dataset(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from mlstudio.training import train
    from PySide6.QtWidgets import QFileDialog
    app,c,b=controller_with_data(tmp_path);f=fixture();c.result=train(f,'target','Decision Tree')
    c.source=None;c.dataset=None
    incoming=tmp_path/'input.csv';output=tmp_path/'pred.csv'
    f.drop(columns='target').head().to_csv(incoming,index=False)
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(incoming),''))
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(output),''))
    c.predict_csv();assert len(pd.read_csv(output))==5;c.close()


def test_portable_snapshot_preserves_infinite_values():
    from mlstudio.project_data import pack_frame,unpack_frame
    f=pd.DataFrame({'x':[1.8267565599574231,np.nan,np.inf,-np.inf],'z':np.random.default_rng(13).normal(size=4),'name':['A','B',None,'D']})
    pd.testing.assert_frame_equal(f,unpack_frame(pack_frame(f)))
