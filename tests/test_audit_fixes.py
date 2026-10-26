import numpy as np
import pandas as pd
import pytest


def fixture():
    f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)), 'target':[0,1]*40})
    f.loc[[2,6,11],'x']=np.nan
    return f


def test_small_knn_has_actionable_preflight():
    from mlstudio.training import train
    f=pd.DataFrame({'x':range(8),'target':[0,1]*4})
    with pytest.raises(ValueError,match='KNN needs.*training rows.*neighbors'):
        train(f,'target','KNN',test_size=.4)


def test_custom_neighbors_and_training_only_search():
    from mlstudio.training import train
    r=train(fixture(),'target','KNN',parameters={'n_neighbors':3},tune=True)
    assert r.metadata['tuning']['folds']==3
    assert r.metadata['test_rows']==20
    assert len(r.metadata['tuning']['candidates'])==3


def test_research_saved_plan_fits_each_fold():
    from mlstudio.research import ResearchConfig,ResearchSession,pipeline_for
    s=ResearchSession(fixture(),ResearchConfig(target='target',preparation={'imputation':'mean','scaling':'minmax','excluded':['z']}))
    assert s.features==['x']
    a,b=s.folds[0]
    p=pipeline_for(s.x.iloc[a],'Decision Tree','baseline',42,s.config.preparation)
    p.fit(s.x.iloc[a],s.y.iloc[a])
    numeric=p.named_steps['preprocess'].named_transformers_['numeric']
    assert numeric.named_steps['imputer'].statistics_[0]==pytest.approx(s.x.iloc[a].x.mean())
    assert numeric.named_steps['scaler'].data_max_[0]==s.x.iloc[a].x.max()
    s.compare(['Decision Tree'])


def test_model_roundtrip_and_untrusted_types(tmp_path):
    from mlstudio.training import train,predict
    from mlstudio.model_io import save_model,load_model
    import skops.io as sio
    f=fixture();r=train(f,'target','Decision Tree',preparation={'outliers':'cap_iqr','feature_selection':'f_test','selection_k':1})
    path=tmp_path/'model.mlmodel';save_model(path,r);loaded=load_model(path)
    np.testing.assert_equal(predict(r,f.drop(columns='target')),predict(loaded,f.drop(columns='target')))
    assert loaded.metadata==r.metadata
    with pytest.raises(ValueError):load_model(tmp_path/'bad.mlmodel')


def test_plot_selection_uses_positions_and_preserves_data():
    from mlstudio.dataset_visuals import select_region
    f=pd.DataFrame({'x':[1.,2.,np.nan,4.],'z':[5.,6.,7.,8.]},index=[1,1,2,2])
    r=select_region(f,{'columns':['x','z'],'xmin':1.5,'xmax':5,'ymin':5,'ymax':7})
    assert r['positions']==[1] and r['count']==1
    assert r['preview']['data'][0]==[2,2.,6.]
    assert len(f)==4


def test_all_registered_models_can_be_reopened(tmp_path):
    from mlstudio.training import train,predict,ALGORITHMS,REGRESSION
    from mlstudio.model_io import save_model,load_model
    for algorithm in ALGORITHMS:
        f=fixture()
        if algorithm in REGRESSION:f['target']=np.arange(80)*.2+np.sin(np.arange(80))
        r=train(f,'target',algorithm);path=tmp_path/'model.mlmodel';save_model(path,r)
        restored=load_model(path)
        np.testing.assert_equal(predict(r,f.drop(columns='target')),predict(restored,f.drop(columns='target')),err_msg=algorithm)


def test_saved_project_contains_data_after_source_removed(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtCore import QEventLoop,QTimer
    app,c,b=controller_with_data(tmp_path)
    source=tmp_path/'source.csv';c.dataset.current.to_csv(source,index=False);c.source=str(source);c.import_options={}
    original=c.dataset.current;project=tmp_path/'portable.json';c.save_project(str(project));source.unlink()
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(project),''))
    c.open_project()
    for _ in range(100):
        if not c.busy:break
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    assert not c.busy
    pd.testing.assert_frame_equal(original,c.dataset.current)
    c.save_project(str(project));c.close()


def test_workflow_preparation_and_comparison_nodes():
    from mlstudio.workflow import Graph
    graph=Graph({'1':{'type':'Dataset'},'2':{'type':'Prepare training'},'3':{'type':'Compare models'}},[['1','2'],['2','3']])
    result=graph.execute(fixture(),'target','Decision Tree',preparation={'scaling':'minmax'})
    assert len(result['3'])==3
    assert all(r.metadata['preparation']['scaling']=='minmax' for r in result['3'])


def test_prediction_import_accepts_non_utf8_comma_decimal(tmp_path,monkeypatch):
    import json
    from test_bridge import controller_with_data
    from mlstudio.training import train
    from PySide6.QtWidgets import QFileDialog
    app,c,b=controller_with_data(tmp_path)
    f=pd.DataFrame({'café':np.arange(24,dtype=float)+.5,'target':[0,1]*12})
    c.result=train(f,'target','Decision Tree')
    incoming=tmp_path/'incoming.csv';outgoing=tmp_path/'predictions.csv'
    f.drop(columns='target').head().to_csv(incoming,index=False,encoding='cp1252',sep=';',decimal=',')
    original=incoming.read_bytes()
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(incoming),''))
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(outgoing),''))
    b.command('predict_configured',json.dumps({'import':{'encoding':'cp1252','decimal':',','delimiter':';','header':True}}))
    result=pd.read_csv(outgoing)
    assert len(result)==5 and 'prediction' in result
    assert result['café'].iloc[0]==.5
    assert incoming.read_bytes()==original
    c.close()
