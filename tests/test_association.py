import pandas as pd
import pytest
from mlstudio.association import mine_rules


def test_exact_support_confidence_lift_with_missing_denominator():
    f=pd.DataFrame({'a':['yes','yes','yes','no',None], 'b':['yes','yes','no','no',None]})
    original=f.copy(deep=True)
    r=mine_rules(f,{'columns':['a','b'],'min_support':.2,'min_confidence':.5})
    rule=next(v for v in r['rules'] if v['antecedent_items']==[{'column':'b','value':'yes','value_type':'str'}])
    assert rule['count']==2 and rule['support']==.4 and rule['confidence']==1
    assert rule['lift']==pytest.approx(5/3)
    pd.testing.assert_frame_equal(f,original)


def test_numeric_requires_explicit_choice_and_limits_validate():
    f=pd.DataFrame({'a':[1,1,0,0],'b':['y','y','n','n']})
    with pytest.raises(ValueError,match='explicit'):mine_rules(f,{'columns':['a','b']})
    assert mine_rules(f,{'columns':['a','b'],'numeric_as_categories':True})['rules']
    with pytest.raises(ValueError,match='0.01'):mine_rules(f,{'columns':['a','b'],'min_support':0,'numeric_as_categories':True})
    with pytest.raises(ValueError,match='50 values'):mine_rules(pd.DataFrame({'a':range(60),'b':['x']*60}),{'columns':['a','b'],'numeric_as_categories':True})


def test_rules_with_three_items_and_no_same_column_contradictions():
    f=pd.DataFrame({'a':['x']*4+['z']*2,'b':['y']*4+['w']*2,'c':['t']*6})
    r=mine_rules(f,{'columns':list(f),'min_support':.2,'max_length':3})
    assert any(len(v['items'])==3 for v in r['frequent_itemsets'])
    assert all(len({i['column'] for i in v['items']})==len(v['items']) for v in r['frequent_itemsets'])


@pytest.mark.parametrize('values',[[['nested'],['value']],['x'*501,'short']])
def test_free_text_and_nested_objects_fail_actionably(values):
    f=pd.DataFrame({'a':values,'b':['x','y']})
    with pytest.raises(ValueError,match='category|categories'):mine_rules(f,{'columns':['a','b']})
