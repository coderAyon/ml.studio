import numpy as np
import pandas as pd
import pytest


def frame():
    f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'label':[0,1]*40})
    f.loc[3,'x']=np.nan
    return f


def graph():
    return {'nodes':{'1':{'type':'Dataset','position':[30,30]},
        '2':{'type':'Impute missing','position':[230,30],'params':{'imputation':'mean'}},
        '3':{'type':'Normalize','position':[430,30],'params':{'scaling':'minmax'}},
        '4':{'type':'Prepared output','position':[630,30]}},'edges':[['1','2'],['2','3'],['3','4']]}


def approve_imputation(engine,g,target='label',algorithm='Decision Tree'):
    g['nodes']['2']['params']['imputation_approval']=engine.imputation_approval(g,'2',target,algorithm)


def test_prepared_output_is_real_training_only_and_source_unchanged():
    from mlstudio.flow_engine import FlowEngine
    f=frame();original=f.copy(deep=True);engine=FlowEngine()
    g=graph();engine.run(g,f,'label','Decision Tree',None,'data-v0')
    approve_imputation(engine,g)
    result=engine.run(g,f,'label','Decision Tree',None,'data-v0')
    r=result['nodes']['4']
    assert result['nodes']['1']['missing_cells']==1
    assert result['nodes']['1']['duplicate_rows']==int(f.duplicated().sum())
    assert r['status']=='ready' and r['kind']=='prepared'
    assert r['training_rows']==60 and r['test_rows']==20
    assert r['preparation']['imputation']=='mean' and r['preparation']['scaling']=='minmax'
    assert r['table']['data'] and 'numeric__x' in r['table']['columns']
    pd.testing.assert_frame_equal(f,original)


def test_disconnected_nodes_wait_and_cache_ignores_positions():
    from mlstudio.flow_engine import FlowEngine
    engine=FlowEngine();g=graph();g['nodes']['5']={'type':'Data table','position':[10,300]}
    one=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert one['nodes']['5']['status']=='waiting'
    approve_imputation(engine,g)
    engine.run(g,frame(),'label','Decision Tree',None,'v0')
    g['nodes']['1']['position']=[600,900]
    two=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert two['computed']==0 and two['cached']==4
    g['nodes']['2']['params']['imputation']='median'
    three=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert three['nodes']['2']['status']=='pending' and three['nodes']['4']['status']=='blocked'
    approve_imputation(engine,g)
    four=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert four['computed']==3 and four['cached']==1


def test_result_branch_error_does_not_block_table_branch_and_cycle_rejected():
    from mlstudio.flow_engine import FlowEngine,validate_flow
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'Train model','position':[200,0]},
        '3':{'type':'Results','position':[400,0]},'4':{'type':'Data table','position':[200,200]}},
        'edges':[['1','2'],['2','3'],['1','4']]}
    result=FlowEngine().run(g,frame(),'','Decision Tree',None,'v0')
    assert result['nodes']['2']['status']=='error' and result['nodes']['3']['status']=='blocked'
    assert result['nodes']['4']['status']=='ready'
    with pytest.raises(ValueError,match='Incompatible'):validate_flow({**g,'edges':[['2','4']]})
    g['nodes']['5']={'type':'Select columns','position':[400,300]};g['edges']=[['1','4'],['4','5'],['5','4']]
    with pytest.raises(ValueError):validate_flow(g)


def test_branch_local_parameters_and_output_model():
    from mlstudio.flow_engine import FlowEngine
    f=frame();g={'nodes':{'1':{'type':'Dataset','position':[0,0]},
        '2':{'type':'Select columns','position':[200,0],'params':{'columns':['x','label']}},
        '3':{'type':'Train model','position':[400,0],'params':{'target':'label','algorithm':'Decision Tree'}},
        '4':{'type':'Results','position':[600,0]}},'edges':[['1','2'],['2','3'],['3','4']]}
    engine=FlowEngine();r=engine.run(g,f,'label','Ridge',None,'v0')
    assert r['nodes']['4']['model']['metadata']['features']==['x']
    assert engine.values['4'].metadata['algorithm']=='Decision Tree'


@pytest.mark.parametrize('kind',['Histogram','Scatter plot','Box plot','Bar chart','Correlation'])
def test_chart_nodes_produce_real_images_and_tables(kind):
    from mlstudio.flow_engine import FlowEngine
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':kind,'position':[200,0]}},'edges':[['1','2']]}
    r=FlowEngine().run(g,frame(),'label','Decision Tree',None,'v0')['nodes']['2']
    assert r['status']=='ready' and r['chart'].startswith('data:image/svg+xml;base64,') and r['table']['data']


def test_exploratory_imputation_guard_blocks_only_evaluation_branch():
    from mlstudio.flow_engine import FlowEngine
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'Train model','position':[200,0]},'3':{'type':'Data table','position':[200,200]}},'edges':[['1','2'],['1','3']]}
    r=FlowEngine().run(g,frame(),'label','Decision Tree',None,'v0',evaluation_error='Undo whole-data filling before evaluation')['nodes']
    assert r['2']['status']=='error' and r['3']['status']=='ready'


def test_project_roundtrip_preserves_node_parameters_and_incomplete_steps(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from mlstudio.workflow import Graph
    from mlstudio.data import DatasetVersions
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtCore import QEventLoop,QTimer
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(frame());source=tmp_path/'data.csv';frame().to_csv(source,index=False);c.source=str(source);c.import_options={};c.refresh_data()
    nodes={'1':{'type':'Dataset','position':[10,20]},'2':{'type':'Normalize','position':[200,20],'params':{'scaling':'minmax'}},'3':{'type':'Results','position':[200,300]}}
    c.canvas.restore(Graph(nodes,[['1','2']]))
    project=tmp_path/'project.json';c.save_project(str(project))
    c.canvas.restore(Graph())
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a:(str(project),''))
    c.open_project()
    for _ in range(100):
        if not c.busy:break
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    assert c.canvas.graph.nodes['2']['params']['scaling']=='minmax'
    assert '3' in c.canvas.graph.nodes and c.canvas.graph.edges==[['1','2']]
    c.close()


def test_prepared_output_does_not_require_an_outcome_for_unsupervised_preview():
    from mlstudio.flow_engine import FlowEngine
    engine=FlowEngine();g=graph();f=frame()
    engine.run(g,f,'','Decision Tree',None,'v0')
    approve_imputation(engine,g,target='')
    r=engine.run(g,f,'','Decision Tree',None,'v0')['nodes']['4']
    assert r['status']=='ready' and r['kind']=='prepared'
    assert r['output_columns']==3 and r['training_rows']==60
    assert 'No outcome selected' in r['note']


def test_connecting_or_forcing_workflow_does_not_impute_without_approval():
    from mlstudio.flow_engine import FlowEngine
    f=frame();original=f.copy(deep=True);engine=FlowEngine();g=graph()
    for clear_cache in (False,False,True):
        if clear_cache:engine.cache.clear()
        r=engine.run(g,f,'label','Decision Tree',None,'v0')['nodes']
        assert r['2']['status']=='pending' and r['2']['kind']=='imputation'
        assert r['2']['missing_cells']==1
        row=r['2']['table']['index'].index(3)
        assert r['2']['table']['data'][row][0] is None
        assert 'comparison' not in r['2']
        assert r['3']['status']=='blocked' and r['4']['status']=='blocked'
        assert '2' not in engine.values and '4' not in engine.values
        pd.testing.assert_frame_equal(f,original)


def test_explicit_imputation_approval_unlocks_real_filled_preview():
    from mlstudio.flow_engine import FlowEngine
    f=frame();original=f.copy(deep=True);engine=FlowEngine();g=graph()
    engine.run(g,f,'label','Decision Tree',None,'v0')
    approve_imputation(engine,g)
    r=engine.run(g,f,'label','Decision Tree',None,'v0')['nodes']
    assert r['2']['status']=='ready' and r['4']['status']=='ready'
    assert r['2']['imputation_preview']['before_missing']==1
    assert r['2']['imputation_preview']['after_missing']==0
    assert not engine.values['2'].imputed_preview['x'].isna().any()
    effect=next(item for item in r['2']['comparison']['effects'] if item['column']=='x')
    assert effect['before_missing']==1 and effect['after_missing']==0
    cell=next(item for item in r['2']['comparison']['cells'] if item['row']==4 and item['column']=='x')
    assert cell['before'] is None and np.isfinite(cell['after'])
    pd.testing.assert_frame_equal(engine.values['2'].frame,original)
    pd.testing.assert_frame_equal(f,original)


@pytest.mark.parametrize('changed',['method','source','upstream','target','algorithm'])
def test_imputation_requires_another_click_after_relevant_changes(changed):
    from mlstudio.flow_engine import FlowEngine
    f=frame();original=f.copy(deep=True);engine=FlowEngine();g=graph()
    target,algorithm,key='label','Decision Tree','v0'
    engine.run(g,f,target,algorithm,None,key);approve_imputation(engine,g)
    assert engine.run(g,f,target,algorithm,None,key)['nodes']['4']['status']=='ready'
    if changed=='method':g['nodes']['2']['params']['imputation']='median'
    elif changed=='source':key='v1'
    elif changed=='upstream':
        g['nodes']['5']={'type':'Select columns','position':[130,30],'params':{'columns':['x','label']}}
        g['edges']=[['1','5'],['5','2'],['2','3'],['3','4']]
    elif changed=='target':target=''
    else:algorithm='Random Forest'
    r=engine.run(g,f,target,algorithm,None,key)['nodes']
    assert r['2']['status']=='pending' and r['4']['status']=='blocked'
    assert 'comparison' not in r['2'] and '4' not in engine.values
    approve_imputation(engine,g,target,algorithm)
    assert engine.run(g,f,target,algorithm,None,key)['nodes']['4']['status']=='ready'
    pd.testing.assert_frame_equal(f,original)


def test_imputation_cannot_be_approved_without_ready_input():
    from mlstudio.flow_engine import FlowEngine
    engine=FlowEngine();g=graph();g['edges']=[]
    engine.run(g,frame(),'label','Decision Tree',None,'v0')
    with pytest.raises(ValueError,match='Connect a completed data step'):
        approve_imputation(engine,g)


@pytest.mark.parametrize('method',['mean','median','most_frequent','constant','knn'])
def test_impute_preview_fills_categories_and_keeps_outcome_and_excluded_cells(method):
    from mlstudio.flow_engine import FlowEngine
    f=frame();f['category']=['a','b']*40;f.loc[71,'category']=None
    f['skip']=np.nan;f.loc[0,'label']=np.nan
    original=f.copy(deep=True);engine=FlowEngine();g=graph()
    g['nodes']['2']['params']['imputation']=method
    prep={'excluded':['skip'],'scaling':'none'}
    engine.run(g,f,'label','Decision Tree',prep,'v0');approve_imputation(engine,g)
    r=engine.run(g,f,'label','Decision Tree',prep,'v0')['nodes']['2']
    assert r['status']=='ready',r
    assert r['imputation_preview']['before_missing']==2
    assert r['imputation_preview']['after_missing']==0
    assert r['imputation_preview']['untouched_missing']==81
    filled=engine.values['2'].imputed_preview
    assert pd.notna(filled.loc[71,'category']) and pd.notna(filled.loc[3,'x'])
    assert pd.isna(filled.loc[0,'label']) and filled['skip'].isna().all()
    assert 71 in r['imputation_preview']['table']['index']
    pd.testing.assert_frame_equal(engine.values['2'].frame,original)
    pd.testing.assert_frame_equal(f,original)
