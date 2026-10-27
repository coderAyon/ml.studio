import numpy as np
import pandas as pd
import pytest
from mlstudio.analyzer import analyze,profile
from mlstudio.data import DatasetVersions,Transformation
from mlstudio.preprocessing import preview_change
from mlstudio.training import train,predict
from mlstudio.flow_engine import FlowEngine


def sample(kind):
    f=pd.DataFrame({'measurement':np.arange(40,dtype=float),'other':np.sin(np.arange(40)),'outcome':['yes','no']*20})
    if kind=='nullable':f['measurement']=pd.Series([1,None,2,3]*10,dtype='Int64')
    elif kind=='text':f=f.drop(columns=['measurement','other']).assign(city=['Dhaka','London',None,'Rome']*10)
    elif kind=='category':f['city']=pd.Categorical(['a','b',None,'a']*10)
    elif kind=='boolean':f['switch']=pd.Series([True,False,None,True]*10,dtype='boolean')
    elif kind=='dates':f['day']=pd.date_range('2026-01-01',periods=40,tz='UTC')
    elif kind=='empty_numeric':f['measurement']=np.nan
    elif kind=='constant':f['measurement']=1.;f['other']=0.
    elif kind=='zero_iqr':f['measurement']=0.;f.loc[35,'measurement']=200
    elif kind=='duplicate_index':f.index=['repeat']*len(f)
    return f


@pytest.mark.parametrize('kind',['numeric','nullable','text','category','boolean','dates','empty_numeric','constant','zero_iqr','duplicate_index'])
@pytest.mark.parametrize('algorithm',['Decision Tree','Logistic Regression'])
def test_supported_data_matrix_profiles_trains_and_predicts_without_mutation(kind,algorithm):
    f=sample(kind);original=f.copy(deep=True)
    assert profile(f)['rows']==40
    assert isinstance(analyze(f),list)
    r=train(f,'outcome',algorithm,preparation={'scaling':'robust'})
    assert len(predict(r,f.drop(columns='outcome').iloc[:3]))==3
    assert r.metadata['training_rows']+r.metadata['test_rows']==40
    pd.testing.assert_frame_equal(f,original)


def test_nullable_integer_median_promotes_type_and_preview_is_truthful():
    f=pd.DataFrame({'x':pd.Series([1,2,None],dtype='Int64')})
    p=preview_change(f,'Fill missing','x')
    assert p['after']['data']==[[1.5]] and p['summary']['changed_cells']==1
    assert p['type_changes'][0]['after']=='Float64'
    assert f.x.isna().sum()==1


def test_category_and_boolean_filling_and_empty_numeric_actionable():
    f=pd.DataFrame({'category':pd.Categorical([None,None])})
    assert Transformation('Fill missing',['category']).preview(f).category.tolist()==['Missing','Missing']
    f=pd.DataFrame({'switch':pd.Series([True,True,None],dtype='boolean')})
    assert Transformation('Fill missing',['switch']).preview(f).switch.tolist()==[True]*3
    with pytest.raises(ValueError,match='entirely missing'):preview_change(pd.DataFrame({'x':[np.nan]*3}),'Fill missing','x')


def test_exclusions_apply_before_retained_feature_limit():
    f=pd.DataFrame({**{f'x{i}':np.arange(40) for i in range(205)},'outcome':['a','b']*20})
    r=train(f,'outcome','Decision Tree',preparation={'excluded':[f'x{i}' for i in range(2,205)]})
    assert r.metadata['features']==['x0','x1']
    with pytest.raises(ValueError,match='200 retained'):train(f,'outcome','Decision Tree')


def test_infinite_outcomes_and_bad_prediction_schemas_explain_recovery():
    f=sample('numeric')
    with pytest.raises(ValueError,match='outcome contains infinite'):train(f.assign(outcome=[np.inf,1]*20),'outcome','Decision Tree')
    r=train(f,'outcome','Decision Tree');x=f.drop(columns='outcome')
    with pytest.raises(ValueError,match='unique'):predict(r,pd.concat([x,x[['measurement']]],axis=1))
    with pytest.raises(ValueError,match='contains infinity'):predict(r,x.assign(measurement=np.inf))


def test_preview_cache_tracks_outcome_and_unsupervised_branches():
    f=sample('numeric');f['outcome']=[0,1]*20
    g={'nodes':{'1':{'type':'Dataset','position':[0,0]},'2':{'type':'Normalize','position':[100,0]}},'edges':[['1','2']]}
    e=FlowEngine();r=e.run(g,f,'outcome','Decision Tree',None,'v0')
    assert 'outcome' not in r['nodes']['2']['comparison']['numeric_columns']
    r=e.run(g,f,'measurement','Ridge',None,'v0')
    assert 'measurement' not in r['nodes']['2']['comparison']['numeric_columns']
    assert 'outcome' in r['nodes']['2']['comparison']['numeric_columns']
    g['nodes']['2']['type']='Select columns';g['nodes']['2']['params']={'columns':['measurement','other']}
    g['nodes']['3']={'type':'Normalize','position':[200,0]};g['edges'].append(['2','3'])
    r=e.run(g,f,'outcome','Decision Tree',None,'v0')
    assert 'preview_error' not in r['nodes']['3']
    assert r['nodes']['3']['comparison']['numeric_columns']==['measurement','other']


def test_selected_rows_preview_apply_undo_and_invalid_positions():
    f=sample('duplicate_index');p=preview_change(f,'Select row positions','', '[1, 12, 35]')
    assert p['summary']['after_rows']==3 and p['summary']['removed_rows']==37
    assert p['aligned_rows'] is False
    v=DatasetVersions(f);v.apply(Transformation(**p['operations'][0]));assert len(v.current)==3
    v.undo();pd.testing.assert_frame_equal(v.current,f)
    with pytest.raises(ValueError):preview_change(f,'Select row positions','', '[80]')
