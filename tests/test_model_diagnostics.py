import numpy as np
import pandas as pd
import pytest
from mlstudio.training import train
from mlstudio.model_diagnostics import build_diagnostics,diagnostics_rows


def dataset():
    return pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'label':[0,1]*40})


def test_classification_diagnostics_use_recorded_holdout_only(monkeypatch):
    f=dataset();model=train(f,'label','Logistic Regression')
    monkeypatch.setattr(model.pipeline,'fit',lambda *a,**k:pytest.fail('Diagnostics must not fit'))
    r=build_diagnostics(model,f,'1')
    assert len(r['charts'])==3 and {'roc','precision_recall','calibration'}=={c['id'] for c in r['charts']}
    assert all(c['image'].startswith('data:image/svg+xml;base64,') for c in r['charts'])
    assert r['holdout_rows']==20 and r['metrics']['Accuracy']==pytest.approx(model.metrics['Accuracy'])
    assert len(diagnostics_rows(r,'predictions'))==20
    positions=r['tables']['predictions']['data'];assert sorted(v[0]-1 for v in positions)==sorted(model.metadata['test_positions'])
    assert 0<=r['metrics']['ROC AUC']<=1
    changed=f.copy();changed.loc[0,'x']=999
    with pytest.raises(ValueError,match='differs'):build_diagnostics(model,changed)


def test_multiclass_selection_and_svm_scores_have_no_fake_calibration():
    f=dataset();f.label=[0,1,2,3]*20;model=train(f,'label','SVM')
    r=build_diagnostics(model,f,'2')
    assert r['positive_class']=='2' and len(r['tables']['per_class']['data'])==4
    assert all(c['id']!='calibration' for c in r['charts'])
    assert any('decision scores' in n for n in r['notes'])
    with pytest.raises(ValueError,match='positive class'):build_diagnostics(model,f,'missing')


def test_regression_residuals_and_tampered_manifest_rejected():
    f=dataset();model=train(f,'x','Ridge');r=build_diagnostics(model,f)
    assert r['task']=='regression' and len(r['charts'])==2
    assert r['metrics']['RMSE']==pytest.approx(model.metrics['RMSE'])
    rows=diagnostics_rows(r,'predictions');assert np.allclose(rows.iloc[:,1]-rows.iloc[:,2],rows.iloc[:,3])
    model.metadata['test_positions']=[0,0]
    with pytest.raises(ValueError,match='manifest'):build_diagnostics(model,f)


def test_column_renames_with_identical_values_cannot_reuse_diagnostics():
    f=dataset();model=train(f,'label','Logistic Regression')
    changed=f.rename(columns={'x':'z','z':'x'})
    with pytest.raises(ValueError,match='schema'):build_diagnostics(model,changed)
    tiny=f[['x','label']];single=train(tiny,'label','Logistic Regression')
    with pytest.raises(ValueError,match='schema'):build_diagnostics(single,tiny.rename(columns={'x':'label','label':'x'}))
    del single.metadata['dataset_columns']
    with pytest.raises(ValueError,match='train a new model'):build_diagnostics(single,tiny)


def test_boolean_to_integer_or_category_domain_change_requires_training_schema():
    f=dataset();f['flag']=[True,False]*40;model=train(f,'label','Logistic Regression')
    changed=f.copy();changed['flag']=changed.flag.astype(int)
    with pytest.raises(ValueError,match='types or category domains'):build_diagnostics(model,changed)
    f['flag']=f.flag.astype('category');categorical=train(f,'label','Logistic Regression')
    changed=f.copy();changed['flag']=changed.flag.cat.rename_categories({True:1,False:0})
    with pytest.raises(ValueError,match='types or category domains'):build_diagnostics(categorical,changed)
