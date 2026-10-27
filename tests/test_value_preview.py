import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlstudio.flow_engine import FlowEngine,Packet


def data():
    f=pd.DataFrame({'value':np.arange(80,dtype=float)%10,'other':np.arange(80,dtype=float),'label':[0,1]*40})
    f.loc[2,'value']=100
    f.index=[7]*len(f)  # Row positions must not depend on unique index labels.
    return f


def test_outlier_before_after_tracks_same_row_and_training_only_scale():
    f=data();original=f.copy(deep=True)
    _,r=FlowEngine().prepared(Packet(f,{'scaling':'minmax'}),'label','Decision Tree',{'preview_row':3,'preview_column':'value'})
    cell=r['comparison']['cells'][0]
    train,test=train_test_split(np.arange(80),test_size=.25,random_state=42,stratify=f.label)
    low=f.value.iloc[train].min();high=f.value.iloc[train].max()
    assert cell['row']==3 and cell['index']=='7' and cell['before']==100
    assert cell['after']==pytest.approx((100-low)/(high-low))
    assert cell['original_outlier'] is True
    assert cell['split']==('Training' if 2 in train else 'Holdout')
    pd.testing.assert_frame_equal(f,original)


def test_normalize_step_shows_comparison_without_changing_downstream_data():
    f=data();g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'Normalize','position':[200,0],
        'params':{'scaling':'minmax','preview_row':3,'preview_column':'value'}}},'edges':[['1','2']]}
    engine=FlowEngine();r=engine.run(g,f,'label','Decision Tree',None,'v0')['nodes']['2']
    assert r['status']=='ready' and r['comparison']['cells'][0]['before']==100
    assert r['comparison']['cells'][0]['after']!=100
    pd.testing.assert_frame_equal(engine.values['2'].frame,f)


def test_default_preview_prioritizes_outliers_including_late_rows():
    f=data();f.iloc[70,0]=200
    _,r=FlowEngine().prepared(Packet(f,{'scaling':'standard'}),'label','Decision Tree')
    cells=r['comparison']['cells']
    assert any(c['row']==71 and c['original_outlier'] for c in cells)
    assert len({c['row'] for c in cells})<=30


@pytest.mark.parametrize('params',[{'preview_row':0},{'preview_row':81},{'preview_row':True},{'preview_column':'label'}])
def test_invalid_preview_selection_rejected(params):
    with pytest.raises(ValueError):FlowEngine().prepared(Packet(data(),{}),'label','Decision Tree',params)


def test_missing_value_preview_and_feature_reduction_keep_numeric_stage_mapping():
    f=data();f.iloc[2,0]=np.nan
    _,r=FlowEngine().prepared(Packet(f,{'feature_selection':'pca','pca_components':1}),
        'label','Decision Tree',{'preview_row':3,'preview_column':'value'})
    cell=r['comparison']['cells'][0]
    assert cell['before'] is None and np.isfinite(cell['after'])
    assert r['output_columns']==1 and cell['column']=='value'
