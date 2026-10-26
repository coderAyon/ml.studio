import json
import pandas as pd
from mlstudio.analyzer import profile


def test_full_row_column_profile_and_empty_data():
    frame=pd.DataFrame({'x':[1.,1.,None,float('inf'),-3.,0.], 'category':['a','a',None,'b','c','d']})
    p=profile(frame);x=p['fields'][0]
    assert (p['rows'],p['columns'],p['total_cells'])==(6,2,12)
    assert p['complete_rows']==5 and p['rows_with_missing']==1 and p['empty_rows']==1
    assert p['duplicate_members']==2
    assert x['present']==5 and x['finite']==4 and x['infinite']==1
    assert x['zero']==1 and x['negative']==1
    assert x['mean']==-.25 and x['q1']==-.75 and x['q3']==1.
    assert x['top_values'][0]['percent']==2/6*100
    json.dumps(p,allow_nan=False)
    empty=profile(pd.DataFrame({'x':pd.Series(dtype=float)}))
    assert empty['fields'][0]['missing_percent']==0
    json.dumps(empty,allow_nan=False)


def test_inspect_row_positions_and_invalid_input(tmp_path):
    from test_bridge import controller_with_data
    from mlstudio.data import DatasetVersions
    app,c,b=controller_with_data(tmp_path)
    frame=pd.DataFrame({'v':pd.Series([pd.NA,3],dtype='Int64'),'text':['<tag>','b']})
    frame.index=[7,22]
    c.dataset=DatasetVersions(frame);c.refresh_data();b.snapshot()
    b.command('inspect_row','{"row":1}')
    row=b.snapshot()['row_detail']
    assert row['position']==1 and row['index']=='7' and row['missing']==1
    assert row['values'][0]['missing'] and not row['values'][0]['infinite']
    assert row['values'][1]['value']=='<tag>'
    b.command('inspect_row','{"row":2}')
    assert b.snapshot()['row_detail']['values'][0]['value']==3
    notices=[];b.notice.connect(lambda n:notices.append(json.loads(n)))
    b.command('inspect_row','{"row":0}')
    assert notices[-1]['kind']=='error'
    assert b.snapshot()['row_detail']['position']==2
    c.dataset=DatasetVersions(frame.head(1));c.refresh_data()
    assert b.snapshot()['row_detail'] is None
    c.close()
