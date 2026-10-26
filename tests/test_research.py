import importlib
import importlib.util
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('mlstudio.research') is not None, 'Research engine is not implemented'
    return importlib.import_module('mlstudio.research')


def fixture():
    rng=np.random.default_rng(42)
    x=rng.normal(size=120)
    frame=pd.DataFrame({'x':x,'z':rng.normal(size=120),'group':np.repeat(np.arange(30),4),'date':pd.date_range('2020-01-01',periods=120).astype(str),'target':(x>0).astype(int)})
    frame.loc[2:8,'z']=np.nan
    return frame


def test_group_and_time_partitions_never_overlap():
    m=api(); frame=fixture()
    cfg=m.ResearchConfig(target='target',protocol='group',group='group',excluded=['date'])
    session=m.ResearchSession(frame,cfg)
    final=set(session.test_positions)
    assert final.isdisjoint(session.development_positions)
    assert set(frame.iloc[list(final)].group).isdisjoint(set(frame.iloc[session.development_positions].group))
    for train,test in session.folds:
        assert set(train).isdisjoint(test) and final.isdisjoint(train) and final.isdisjoint(test)
        assert set(frame.iloc[train].group).isdisjoint(frame.iloc[test].group)
    time_session=m.ResearchSession(frame,m.ResearchConfig(target='target',protocol='time',time_column='date',excluded=['group']))
    assert max(time_session.development_positions)<min(time_session.test_positions)
    for tr,va in time_session.folds:
        assert max(tr)<min(va)
    assert 'date' not in time_session.features
    assert 'group' not in session.features


def test_reliability_uses_identical_folds_and_keeps_final_untouched():
    m=api(); session=m.ResearchSession(fixture(),m.ResearchConfig(target='target',excluded=['group','date']))
    result=session.compare(['Logistic Regression','Decision Tree'])
    assert len(result['models'])==2
    assert all(len(row['scores'])==6 for row in result['models'])
    assert not session.final_used
    assert result['models'][0]['fold_ids']==result['models'][1]['fold_ids']
    assert len(result['baseline']['scores'])==6
    for tr,va in session.folds:
        pipeline=m.pipeline_for(session.x.iloc[tr],'Logistic Regression','baseline',42)
        pipeline.fit(session.x.iloc[tr],session.y.iloc[tr])
        fitted=pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer'].statistics_[1]
        assert fitted==pytest.approx(session.x.iloc[tr].z.median())
        break
    a=session.evaluate_final('Logistic Regression','baseline')
    assert len(a['predictions'])==len(session.test_positions)
    assert a['metrics']['Accuracy']>=.8
    with pytest.raises(ValueError,match='already'):
        session.evaluate_final('Decision Tree','baseline')


def test_impact_reports_paired_changes_and_final_isolation():
    m=api(); session=m.ResearchSession(fixture(),m.ResearchConfig(target='target',excluded=['group','date']))
    result=session.impact('Logistic Regression')
    assert {r['variant'] for r in result['variants']}=={'baseline','mean','unscaled','indicator'}
    for r in result['variants']:
        assert len(r['scores'])==6
        assert len(r['paired_differences'])==6
    assert result['variants'][0]['mean_difference']==0
    assert session.final_used is False


def test_safety_reports_evidence_without_claiming_certain_leakage():
    m=api(); frame=fixture();frame['copied_target']=frame.target
    issues=m.safety_advice(frame,m.ResearchConfig(target='target'))
    codes={i['code'] for i in issues}
    assert 'target_copy' in codes
    assert 'possible_group' in codes
    assert 'possible_time' in codes
    copied=next(i for i in issues if i['code']=='target_copy')
    assert copied['status']=='Potential issue'
    assert copied['columns']==['copied_target']


def test_error_detective_counts_real_rows_and_regression_errors():
    m=api(); from mlstudio.error_analysis import error_report
    frame=pd.DataFrame({'age':[10,20,30,40], 'x':[None,1,2,3]})
    report=error_report(frame,[0,1,2,3],[0,0,1,1],[0,1,1,0],'classification')
    assert report['error_count']==2
    assert {r['position'] for r in report['rows'] if r['error']}=={1,3}
    assert report['slices']['missingness'][0]['count']+report['slices']['missingness'][1]['count']==4
    reg=error_report(frame,[1,3],[10.,20.],[13.,18.],'regression')
    assert reg['mae']==2.5
    assert reg['rows'][0]['absolute_error']==3


def test_invalid_configs_and_rare_classes_fail_clearly():
    m=api(); frame=fixture()
    with pytest.raises(ValueError):m.ResearchSession(frame,m.ResearchConfig(target='target',protocol='group',group='target'))
    with pytest.raises(ValueError):m.ResearchSession(frame,m.ResearchConfig(target='target',protocol='time',time_column='x'))
    frame['target']=[0]*119+[1]
    with pytest.raises(ValueError):m.ResearchSession(frame,m.ResearchConfig(target='target',excluded=['group','date']))


def test_research_pack_replays_metrics_and_validates_fingerprint(tmp_path):
    m=api(); from mlstudio.research_export import export_pack
    session=m.ResearchSession(fixture(),m.ResearchConfig(target='target',excluded=['group','date']))
    session.compare(['Logistic Regression'])
    session.impact('Logistic Regression')
    session.evaluate_final('Logistic Regression','baseline')
    dest=tmp_path/'research.zip'
    export_pack(dest,session,{'source':'synthetic.csv','transformations':[]})
    out=tmp_path/'replay'
    with zipfile.ZipFile(dest) as archive:archive.extractall(out)
    assert (out/'dataset.csv').exists() and (out/'splits.json').exists()
    process=subprocess.run([sys.executable,'reproduce.py'],cwd=out,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    reproduced=json.loads((out/'reproduced.json').read_text())
    assert reproduced['verified'] is True
    with (out/'dataset.csv').open('a') as handle:handle.write('\n')
    broken=subprocess.run([sys.executable,'reproduce.py'],cwd=out,capture_output=True,text=True,timeout=30)
    assert broken.returncode!=0 and 'fingerprint' in broken.stderr.lower()


def test_final_exposure_is_persisted_across_store_reopening(tmp_path):
    from mlstudio.storage import Store
    store=Store(tmp_path)
    assert hasattr(store,'record_final_exposure'), 'Final-test exposure audit is missing'
    store.record_final_exposure('abc','split-a')
    assert Store(tmp_path).final_exposure_count('abc')==1
    assert Store(tmp_path).final_exposure_count('different')==0


def test_time_ties_never_cross_boundaries_and_regression_replays(tmp_path):
    m=api();from mlstudio.research_export import export_pack
    frame=fixture();frame['date']=np.repeat(pd.date_range('2020-01-01',periods=30).astype(str),4)
    frame['amount']=frame.x*3+2
    session=m.ResearchSession(frame,m.ResearchConfig(target='amount',task='regression',protocol='time',time_column='date',excluded=['group','target']))
    for train,valid in session.folds:
        assert pd.to_datetime(frame.iloc[train].date).max()<pd.to_datetime(frame.iloc[valid].date).min()
    result=session.compare(['Linear Regression','Ridge'])
    assert result['models'][0]['mean']<1e-8
    final=session.evaluate_final('Linear Regression','baseline')
    assert final['metrics']['MAE']<1e-8
    assert final['errors']['task']=='regression'


def test_error_slices_handle_nullable_constant_numeric():
    api();from mlstudio.error_analysis import error_report
    frame=pd.DataFrame({'empty':pd.Series([None,1,None,1],dtype='Int64'),'text':['a','b','b','a']})
    result=error_report(frame,[0,1,2,3],[0,0,1,1],[0,1,1,0],'classification')
    assert result['total']==4
