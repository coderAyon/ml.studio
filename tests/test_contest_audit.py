import json
import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtWidgets import QFileDialog
from test_bridge import controller_with_data
from mlstudio.data import DatasetVersions,Transformation
from mlstudio.analyzer import analyze
from mlstudio.workflow import Graph
from mlstudio.training import train


def test_errors_are_reported_without_hidden_native_modal(tmp_path):
    app,c,b=controller_with_data(tmp_path)
    notices=[];b.notice.connect(lambda s:notices.append(json.loads(s)))
    c.guard(lambda:(_ for _ in ()).throw(ValueError('invalid settings')))
    assert notices[-1]['message']=='invalid settings'
    c.background(lambda:(_ for _ in ()).throw(ValueError('bad training input')),lambda x:None,'Testing error recovery')
    loop=QEventLoop();timer=QTimer();timer.timeout.connect(lambda:loop.quit() if not c.busy else None);timer.start(10);QTimer.singleShot(5000,loop.quit);loop.exec();timer.stop()
    assert not c.busy and notices[-1]['message']=='bad training input'
    assert b.snapshot()['dataset']['rows']==3
    c.close()


def test_empty_versions_and_bulk_plan_are_rejected_atomically(tmp_path):
    app,c,b=controller_with_data(tmp_path)
    before=c.dataset.current
    with pytest.raises(ValueError,match='no data'):c.dataset.apply(Transformation('Drop columns',list(before)))
    with pytest.raises(ValueError):c.dataset.apply(Transformation('Filter rows',['name'],'nonexistent'))
    b.command('bulk_preview',json.dumps({'operations':[{'operation':'Trim whitespace','columns':['name']},{'operation':'Drop columns','columns':list(before)}]}))
    assert b.preview is None and c.dataset.index==0
    pd.testing.assert_frame_equal(c.dataset.current,before)
    with pytest.raises(ValueError,match='Dataset'):Graph().order()
    c.close()


def test_infinity_can_be_inspected_and_corrected_without_ui_failure(tmp_path):
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(pd.DataFrame({'x':[1.,np.inf,-np.inf,np.nan],'category':['a']*4}))
    c.issues=analyze(c.dataset.current)
    s=b.snapshot()
    assert any(i['title']=='Infinite numeric values' for i in s['issues'])
    json.dumps(s,allow_nan=False,default=str)
    c.dataset.apply(Transformation('Replace infinity with missing',['x']))
    assert c.dataset.current.x.isna().sum()==3
    c.dataset.undo();assert np.isinf(c.dataset.current.x.iloc[1])
    c.close()


def test_exports_cannot_overwrite_source_project_or_history(tmp_path,monkeypatch):
    app,c,b=controller_with_data(tmp_path)
    source=tmp_path/'source.csv';source.write_text('x,name\n1,a\n');c.source=str(source)
    project=tmp_path/'project.json';project.write_text('{}');c.active_project_path=str(project)
    for output in [source,project,c.store.path]:
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,p=output,**k:(str(p),''))
        for method in [c.report,c.export_log,c.export_csv]:
            with pytest.raises(ValueError,match='separate output'):method()
    assert source.read_text()=='x,name\n1,a\n' and project.read_text()=='{}'
    c.close()


def test_training_input_budget_and_task_errors_are_clear():
    frame=pd.DataFrame({'x':range(80),'target':np.linspace(.1,3.3,80)})
    with pytest.raises(ValueError,match='Regression'):train(frame,'target','Decision Tree')
    frame['target']=[0]*79+[1]
    with pytest.raises(ValueError,match='two rows'):train(frame,'target','Decision Tree')
    frame['target']=[0,1]*40;frame['x']=frame.x.astype(float);frame.loc[0,'x']=np.inf
    with pytest.raises(ValueError,match='infinity'):train(frame,'target','Decision Tree')
    with pytest.raises(ValueError,match='8–20,000'):train(pd.DataFrame({'x':range(4),'target':[0,1]*2}),'target','Decision Tree')
    with pytest.raises(ValueError,match='memory budget'):train(pd.DataFrame({'identifier':[f'row-{i}' for i in range(2300)],'target':[0,1]*1150}),'target','Decision Tree')


def test_project_save_does_not_replace_unrelated_temp_file(tmp_path):
    app,c,b=controller_with_data(tmp_path)
    source=tmp_path/'project.tmp';source.write_text('x,name\n1,a\n')
    c.source=str(source);c.import_options={'delimiter':',','encoding':'utf-8-sig','header':True,'decimal':'.'}
    c.save_project(str(tmp_path/'project.json'))
    assert source.read_text()=='x,name\n1,a\n'
    assert (tmp_path/'project.json').is_file()
    c.close()


def test_prediction_export_preserves_feature_named_prediction(tmp_path,monkeypatch):
    app,c,b=controller_with_data(tmp_path)
    training=pd.DataFrame({'prediction':range(80),'target':[0,1]*40})
    c.result=train(training,'target','Decision Tree')
    source=tmp_path/'training.csv';training.to_csv(source,index=False);c.source=str(source)
    incoming=tmp_path/'incoming.csv';training[['prediction']].head(10).to_csv(incoming,index=False)
    output=tmp_path/'predicted.csv'
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(incoming),''))
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(output),''))
    c.predict_csv();exported=pd.read_csv(output)
    assert list(exported)==['prediction','_prediction']
    assert exported.prediction.tolist()==list(range(10))
    c.close()
