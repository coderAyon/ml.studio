import pandas as pd
import pytest
from mlstudio.table_operations import combine_tables, summarize_table
from mlstudio.flow_engine import FlowEngine
from mlstudio.workflow import Graph


def test_merge_guards_keys_and_preserves_inputs():
    a=pd.DataFrame({'id':[1,2,2],'x':[3,4,5]});b=pd.DataFrame({'key':[2,3],'x':[8,9]})
    original=a.copy();out=combine_tables(a,b,'Merge data',{'left_key':'id','right_key':'key'})
    assert len(out)==3 and out.x.tolist()==[3,4,5] and out.x_extra.iloc[1]==8
    assert pd.isna(out.x_extra.iloc[0]);pd.testing.assert_frame_equal(a,original)
    with pytest.raises(ValueError,match='repeated'):combine_tables(a,pd.concat([b,b]),'Merge data',{'left_key':'id','right_key':'key'})
    a.loc[0,'id']=None
    with pytest.raises(ValueError,match='missing'):combine_tables(a,b,'Merge data',{'left_key':'id','right_key':'key'})


def test_concatenate_columns_types_and_limits():
    a=pd.DataFrame({'x':[1,2],'a':['a','b']});b=pd.DataFrame({'x':[3],'b':['c']})
    out=combine_tables(a,b,'Concatenate',{});assert len(out)==3 and list(out)==['x','a','b']
    assert list(combine_tables(a,b,'Concatenate',{'columns':'intersection'}))==['x']
    with pytest.raises(ValueError,match='different'):combine_tables(a,pd.DataFrame({'x':['str']}),'Concatenate',{})
    with pytest.raises(ValueError,match='20,000'):combine_tables(pd.DataFrame({'x':range(15000)}),pd.DataFrame({'x':range(10000)}),'Concatenate',{})


def test_group_pivot_values_and_missing_groups():
    f=pd.DataFrame({'g':['a','a','b',None],'c':['u','v','u','v'],'v':[1,3,8,2]})
    out=summarize_table(f,'Group by',{'group':'g','value':'v','statistic':'mean'})
    assert out.iloc[0,1]==2 and len(out)==3
    pivot=summarize_table(f,'Pivot table',{'group':'g','column':'c','statistic':'count'})
    assert pivot['c=u'].sum()==2 and pivot['c=v'].sum()==2


def test_two_input_flow_cache_and_missing_second_input():
    a=pd.DataFrame({'id':[1,2],'x':[4,5]});b=pd.DataFrame({'id':[2,3],'z':[8,9]})
    g={'nodes':{'1':{'type':'Dataset','position':[0,0],'params':{'source':'file','path':'a.csv'}},
        '2':{'type':'Dataset','position':[0,200],'params':{'source':'file','path':'b.csv'}},
        '3':{'type':'Merge data','position':[200,0],'params':{'left_key':'id','right_key':'id'}},
        '4':{'type':'Data table','position':[400,0]}},'edges':[['1','3'],['2','3'],['3','4']]}
    engine=FlowEngine();run=lambda key:engine.run(g,None,'','Decision Tree',None,{'workspace':None,'files':{'1':'a','2':key}},sources={'1':a,'2':b},strict_sources=True)
    r=run('b');assert r['nodes']['4']['status']=='ready' and engine.values['4'].frame.z.iloc[1]==8
    assert run('b')['cached']==4
    b.loc[0,'z']=10;assert run('b2')['computed']==3 and engine.values['4'].frame.z.iloc[1]==10
    g['edges'].remove(['2','3']);assert run('b2')['nodes']['3']['status']=='waiting'
    with pytest.raises(ValueError,match='two tables'):Graph(g['nodes'],g['edges']).order()


def test_summary_branch_blocks_both_evaluation_engines():
    f=pd.DataFrame({'g':['a','b']*40,'x':range(80)})
    nodes={'1':{'type':'Dataset','position':[0,0]},
        '2':{'type':'Group by','position':[200,0],'params':{'group':'g'}},
        '3':{'type':'Train model','position':[400,0]}}
    edges=[['1','2'],['2','3']]
    r=FlowEngine().run({'nodes':nodes,'edges':edges},f,'Row count','Decision Tree',None,'v0')
    assert r['nodes']['3']['status']=='error' and 'summarizes' in r['nodes']['3']['message']
    with pytest.raises(ValueError,match='before grouping'):Graph(nodes,edges).execute(f,'Row count','Decision Tree')
