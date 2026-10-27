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


def test_prepared_output_is_real_training_only_and_source_unchanged():
    from mlstudio.flow_engine import FlowEngine
    f=frame();original=f.copy(deep=True);engine=FlowEngine()
    result=engine.run(graph(),f,'label','Decision Tree',None,'data-v0')
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
    g['nodes']['1']['position']=[600,900]
    two=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert two['computed']==0 and two['cached']==4
    g['nodes']['2']['params']['imputation']='median'
    three=engine.run(g,frame(),'label','Decision Tree',None,'v0')
    assert three['computed']==3 and three['cached']==1


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
    r=FlowEngine().run(graph(),frame(),'','Decision Tree',None,'v0')['nodes']['4']
    assert r['status']=='ready' and r['kind']=='prepared'
    assert r['output_columns']==3 and r['training_rows']==60
    assert 'No outcome selected' in r['note']
