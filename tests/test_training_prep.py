import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from mlstudio.training import train,predict
from mlstudio.training_prep import FoldIQRCapper,validate_preparation,assert_evaluation_ready
from mlstudio.data import DatasetVersions,Transformation


def test_capper_does_not_learn_test_extremes_or_change_input():
    x=np.array([[1.,np.nan],[2.,3.],[3.,3.],[4.,3.],[5.,3.]])
    original=x.copy();cap=FoldIQRCapper().fit(x)
    assert cap.upper_[0]==7.
    assert np.isinf(cap.upper_[1])  # Zero-IQR columns keep valid extremes.
    transformed=cap.transform([[1000.,200.]])
    assert transformed.tolist()==[[7.,200.]]
    np.testing.assert_equal(x,original)


def test_training_preparation_fit_only_train_and_prediction_schema():
    frame=pd.DataFrame({'x':[float(i) for i in range(48)],'id':list(range(48)),
                        'category':['a','b']*24,'target':[0,1]*24})
    frame.loc[3,'x']=np.nan;frame.loc[47,'x']=100000.
    xtrain,_,_,_=train_test_split(frame.drop(columns=['target','id']),frame.target,random_state=42,test_size=.25,stratify=frame.target)
    expected=FoldIQRCapper().fit(xtrain[['x']])
    result=train(frame,'target','Decision Tree',preparation={'outliers':'cap_iqr','scaling':'robust','excluded':['id']})
    pipeline=result.pipeline.named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(pipeline.named_steps['outliers'].upper_,expected.upper_)
    median=np.nanmedian(expected.transform(xtrain[['x']]),axis=0)
    np.testing.assert_allclose(pipeline.named_steps['imputer'].statistics_,median)
    assert result.metadata['features']==['x','category']
    assert len(predict(result,frame[['x','category']].head(2)))==2
    with pytest.raises(ValueError,match='schema|match'):predict(result,frame[['x','category','id']].head(2))


def test_invalid_preparation_and_exploratory_guard():
    for bad in [{'scaling':'magical'},{'excluded':['missing']},{'excluded':['target']},{'outliers':'delete'}]:
        with pytest.raises(ValueError):validate_preparation(bad,['x','target'],'target')
    versions=DatasetVersions(pd.DataFrame({'x':[1.,2.,3.,4.,100.]}))
    versions.apply(Transformation('Cap outliers (IQR)',['x']))
    with pytest.raises(ValueError,match='Undo'):assert_evaluation_ready(versions)
    versions.undo();assert_evaluation_ready(versions)


def test_preprocessing_bridge_exact_rows_apply_undo_and_stale_preview(tmp_path):
    import json
    from test_bridge import controller_with_data
    from mlstudio.analyzer import analyze
    from PySide6.QtCore import QEventLoop,QTimer
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(pd.DataFrame({'x':[1.,2.,3.,4.]*20+[100.],'category':['a']*81}))
    c.issues=analyze(c.dataset.current);c.refresh_data();b.snapshot()
    issue=next(i for i in c.issues if i.title=='Potential outliers')
    b.command('preprocess_inspect',json.dumps({'id':issue.id}))
    detail=b.snapshot()['preprocessing']['detail']
    assert detail['preview']['positions']==[81]
    assert detail['preview']['data'][0][0]==100.
    b.command('preprocess_preview',json.dumps({'operation':'Cap outliers (IQR)','column':'x'}))
    preview=b.snapshot()['preprocessing']['preview']
    assert preview['summary']['changed_cells']==1
    assert c.dataset.index==0
    b.command('preprocess_apply')
    assert c.dataset.index==1
    loop=QEventLoop();QTimer.singleShot(400,loop.quit);loop.exec()
    assert b.snapshot()['preprocessing']['preview'] is None
    c.dataset.undo();c.issues=analyze(c.dataset.current);c.refresh_data();b.snapshot()
    assert c.dataset.current.x.iloc[-1]==100.
    b.command('training_preparation','{"imputation":"mean","scaling":"none","outliers":"cap_iqr","excluded":[]}')
    assert c.preparation['outliers']=='cap_iqr' and c.dataset.index==0
    c.close()


def test_nondefault_training_preparation_survives_project_reopen(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtCore import QEventLoop,QTimer
    app,c,b=controller_with_data(tmp_path)
    source=tmp_path/'data.csv';c.dataset.current.to_csv(source,index=False)
    c.import_options={}
    c.preparation={'imputation':'mean','scaling':'robust','outliers':'cap_iqr','excluded':['name']}
    dest=tmp_path/'project.json';c.save_project(str(dest))
    c.preparation=None
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(dest),''))
    c.open_project()
    for _ in range(100):
        if not c.busy:break
        loop=QEventLoop();QTimer.singleShot(25,loop.quit);loop.exec()
    assert not c.busy
    assert c.preparation==validate_preparation({'imputation':'mean','scaling':'robust','outliers':'cap_iqr','excluded':['name']})
    c.close()
