import numpy as np
import pandas as pd
import pytest
from mlstudio.flow_engine import FlowEngine,Packet
from mlstudio.training import train
from mlstudio.result_brief import result_brief,brief_html


def data():
    f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'label':[0,1]*40});f.loc[3,'x']=np.nan
    return f


def test_preparation_effects_cover_all_rows_using_training_only_statistics():
    f=data();_,output=FlowEngine().prepared(Packet(f,{'scaling':'minmax'}),'label','Decision Tree')
    effect=output['comparison']['effects'][0]
    assert effect['before_missing']==1 and effect['after_missing']==0
    assert effect['before_min']==0 and effect['before_max']==79 and effect['changed']>1
    assert output['comparison']['training_rows']==60 and output['comparison']['test_rows']==20
    assert f.x.isna().sum()==1


def test_result_brief_uses_fitted_evidence_and_escapes_user_supplied_labels():
    f=data();r=train(f,'label','Decision Tree');brief=result_brief(r)
    assert brief['score']==r.metrics['Accuracy'] and brief['test_rows']==20
    assert brief['baseline']==r.metadata['diagnostics']['baseline']['value']
    assert brief['improvement']==pytest.approx(brief['score']-brief['baseline'])
    r.metadata['target']='<script>alert(1)</script>'
    exported=brief_html(r)
    assert '<script>' not in exported and '&lt;script&gt;' in exported
    with pytest.raises(ValueError):result_brief(None)


def test_regression_brief_handles_lower_is_better_and_missing_baseline():
    f=data().assign(outcome=np.arange(80)*2.);r=train(f,'outcome','Ridge');b=result_brief(r)
    assert b['metric']=='RMSE' and b['improvement']==pytest.approx(b['baseline']-b['score'])
    r.metadata.pop('diagnostics');assert result_brief(r)['improvement'] is None


def test_export_guard_protects_every_workflow_csv_source(tmp_path):
    from test_bridge import controller_with_data
    app,c,b=controller_with_data(tmp_path);source=tmp_path/'independent.csv';data().to_csv(source,index=False)
    c.canvas.graph.nodes={'1':{'type':'Dataset','position':[0,0],'params':{'source':'file','path':str(source)}}}
    with pytest.raises(ValueError,match='separate output'):c.protect_output(source)
