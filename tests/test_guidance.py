import pandas as pd
from mlstudio import guidance


def test_automatic_visuals_use_real_values_and_missing_categories():
    frame = pd.DataFrame({'x':[1.,2.,3.,None], 'y':[2.,4.,6.,8.], 'group':['a','a','b',None]})
    visuals = guidance.visual_summary(frame)
    assert sum(visuals['histogram']['counts']) == 3
    assert len(visuals['scatter']['points']) == 3
    assert dict(zip(visuals['bar']['labels'], visuals['bar']['counts']))['(missing)'] == 1
    assert visuals['rows'] == 4
    assert frame.x.isna().sum() == 1


def test_plan_depends_on_observed_data_and_chosen_target():
    frame = pd.DataFrame({'x':[1.,None,3.,4.,5.,6.,7.,8.], 'label':[' A','a','b','b','a','b','a','b'], 'constant':[1]*8, 'target':[0,1]*4})
    plan = guidance.preprocessing_plan(frame, 'target', 'classification')
    steps = plan['steps']
    assert any(s['operation']=='Trim whitespace' and s['columns']==['label'] for s in steps)
    assert any('training' in s['reason'].lower() and s['columns']==['x'] for s in steps)
    assert not any(s.get('operation')=='Drop columns' and 'target' in s['columns'] for s in steps)
    assert plan['task']=='classification'
    assert 'small' in ' '.join(plan['warnings']).lower()
    assert any(a['algorithm']=='Logistic Regression' for a in plan['algorithms'])


def test_comparison_explains_ties_and_never_fabricates_a_winner():
    records = [{'algorithm':'Decision Tree', 'result':{'metrics':{'F1 (weighted)':.8,'Accuracy':.9}, 'metadata':{'seconds':1}}}, {'algorithm':'Random Forest','result':{'metrics':{'F1 (weighted)':.8,'Accuracy':.95}, 'metadata':{'seconds':2}}}]
    explanation = guidance.explain_comparison(records, 'classification')
    assert explanation['leaders'] == ['Decision Tree','Random Forest']
    assert explanation['metric']=='F1 (weighted)'
    assert 'holdout' in explanation['caution'].lower()
    assert guidance.explain_comparison([], 'classification')['leaders']==[]
