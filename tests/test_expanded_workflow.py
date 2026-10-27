import numpy as np
import pandas as pd
import pytest
from mlstudio.flow_engine import FlowEngine,CATALOG,LEARNERS


def data():
    rng=np.random.default_rng(42)
    f=pd.DataFrame({'x':rng.normal(size=120),'z':rng.normal(size=120),'label':[0,1]*60,'category':[' A ','B']*60,'number':['12','15']*60})
    f.loc[3,'x']=np.nan;f.loc[59,'z']=200
    return f


PARAMS={'Filter rows':{'column':'category','value':'A'},'Sample rows':{'rows':30},'Drop missing rows':{'column':'x'},
    'Rename column':{'column':'category','value':'place'},'Convert numeric':{'column':'number'},'Trim whitespace':{'column':'category'},
    'PCA':{'components':2},'Select features':{'k':2},'Discretize':{'bins':4},'Statistics':{},'Outlier review':{},
    'K-Means':{'columns':['x','z'],'clusters':2},'DBSCAN':{'columns':['x','z'],'eps':.5,'min_samples':3},'Hierarchical':{'columns':['x','z'],'clusters':2}}


@pytest.mark.parametrize('kind',list(PARAMS)+sorted(LEARNERS))
def test_every_added_tool_computes_real_outputs_without_mutating_source(kind):
    f=data();original=f.copy(deep=True)
    g={'nodes':{'1':{'type':'Dataset','position':[0,0],'params':{'source':'workspace'}},
        '2':{'type':kind,'position':[150,0],'params':PARAMS.get(kind,{})}},'edges':[['1','2']]}
    engine=FlowEngine();r=engine.run(g,f,'label','Decision Tree',None,'v0',strict_sources=True)['nodes']['2']
    assert r['status']=='ready',r
    if kind=='Sample rows':assert len(engine.values['2'].frame)==30
    if kind=='Filter rows':assert len(engine.values['2'].frame)==60
    if kind=='Drop missing rows':assert len(engine.values['2'].frame)==119
    if kind=='Trim whitespace':assert engine.values['2'].frame.category.iloc[0]=='A'
    if kind in {'K-Means','DBSCAN','Hierarchical'}:assert 'Cluster' in engine.values['2'].frame and len(engine.values['2'].frame)==120
    if kind in LEARNERS:assert r['model']['metadata']['algorithm']==kind
    if kind in {'PCA','Select features','Discretize'}:assert r['comparison']['cells']
    pd.testing.assert_frame_equal(f,original)


def test_empty_import_does_not_implicitly_use_loaded_workspace_and_sources_are_independent():
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'Dataset','position':[0,150],'params':{'source':'file','path':'b.csv'}}},'edges':[]}
    a=data();b=data().iloc[:40].copy()
    engine=FlowEngine();r=engine.run(g,a,'label','Decision Tree',None,'v0',sources={'2':b},strict_sources=True)
    assert r['nodes']['1']['status']=='waiting' and r['nodes']['2']['rows']==40
    g['nodes']['1']['params']={'source':'workspace'}
    r=engine.run(g,a,'label','Decision Tree',None,'v0',sources={'2':b},strict_sources=True)
    assert r['nodes']['1']['rows']==120 and r['nodes']['2']['rows']==40
    assert len(CATALOG)==38 and len({row[0] for row in CATALOG})==38


def test_dirty_workspace_does_not_block_an_independent_fresh_csv_branch():
    g={'nodes':{'1':{'type':'Dataset','position':[0,0],'params':{'source':'file','path':'fresh.csv'}},'2':{'type':'Decision Tree','position':[150,0]}},'edges':[['1','2']]}
    r=FlowEngine().run(g,data(),'label','Decision Tree',None,'v0',evaluation_error='Undo whole-data imputation',sources={'1':data()},strict_sources=True)
    assert r['nodes']['2']['status']=='ready'


def test_large_unused_workspace_does_not_block_a_small_independent_file():
    g={'nodes':{'1':{'type':'Dataset','position':[0,0],'params':{'source':'file','path':'small.csv'}},'2':{'type':'Dataset','position':[0,150],'params':{'source':'workspace'}}},'edges':[]}
    r=FlowEngine().run(g,pd.DataFrame({'x':range(20001)}),'label','Decision Tree',None,'v0',sources={'1':data()},strict_sources=True)
    assert r['nodes']['1']['rows']==120 and r['nodes']['2']['status']=='error'


def test_cluster_labels_can_flow_to_tables_but_do_not_silently_enter_model_evaluation():
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'K-Means','position':[100,0],'params':{'columns':['x','z']}},'3':{'type':'Data table','position':[200,0]},'4':{'type':'Decision Tree','position':[200,150]}},'edges':[['1','2'],['2','3'],['2','4']]}
    r=FlowEngine().run(g,data(),'label','Decision Tree',None,'v0')['nodes']
    assert r['3']['status']=='ready' and 'Cluster' in r['3']['fields']
    assert r['4']['status']=='error' and 'before clustering' in r['4']['message']


def test_file_cache_invalidation_preserves_an_unrelated_branch():
    g={'nodes':{'1':{'type':'Dataset','position':[0,0],'params':{'source':'file','path':'one.csv'}},'2':{'type':'Dataset','position':[0,100],'params':{'source':'file','path':'two.csv'}}},'edges':[]}
    engine=FlowEngine();a=data();b=data().iloc[:40]
    key={'workspace':'v0','files':{'1':'a0','2':'b0'}}
    engine.run(g,a,'label','Decision Tree',None,key,sources={'1':a,'2':b},strict_sources=True)
    key['files']['1']='a1'
    r=engine.run(g,a,'label','Decision Tree',None,key,sources={'1':a.iloc[:70],'2':b},strict_sources=True)
    assert r['computed']==1 and r['cached']==1
    assert r['nodes']['1']['rows']==70 and r['nodes']['2']['rows']==40
