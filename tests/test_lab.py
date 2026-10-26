import json
import numpy as np
import pandas as pd
import pytest
from mlstudio.lab import grouped_analysis


def fixture():
    return pd.DataFrame({'species':['A','A','B','B','C','C'],
                         'sepal_length':[2.,4.,6.,8.,10.,12.],
                         'sepal_width':[1.,3.,5.,7.,9.,11.],
                         'petal_length':[3.,5.,7.,9.,11.,13.],
                         'petal_width':[4.,6.,8.,10.,12.,14.]})


def test_three_lab_tasks_with_real_values():
    df=fixture()
    one,fig=grouped_analysis(df,'species',['sepal_length'],title='Average sepal length',xlabel='Species',ylabel='Average length')
    assert one['table']['data']==[['A',3.],['B',7.],['C',11.]]
    assert [p.get_height() for p in fig.axes[0].patches]==[3.,7.,11.]
    assert fig.axes[0].get_xlabel()=='Species'
    two,countfig=grouped_analysis(df,'species',statistic='count')
    assert two['table']['data']==[['A',2.],['B',2.],['C',2.]]
    three,subplots=grouped_analysis(df,'species',list(df.columns[1:]),layout='subplots')
    assert len(subplots.axes)==3
    assert [a.get_title() for a in subplots.axes]==['A','B','C']
    assert [p.get_height() for p in subplots.axes[0].patches]==[3.,2.,4.,5.]
    pd.testing.assert_frame_equal(df,fixture())


def test_missing_infinite_values_and_full_data():
    df=pd.DataFrame({'g':['A','A',None,None],'v':[np.inf,4.,np.nan,6.]})
    result,_=grouped_analysis(df,'g',['v'])
    assert result['table']['data']==[['A',4.],['(missing group)',6.]]
    counts,_=grouped_analysis(df,'g',statistic='count')
    assert counts['table']['data'][1][1]==2
    missing,_=grouped_analysis(pd.DataFrame({'g':['A'],'v':[np.nan]}),'g',['v'],statistic='sum')
    assert missing['table']['data']==[['A',None]]
    json.dumps(missing,allow_nan=False)
    large=pd.DataFrame({'g':['A']*10000+['B'],'v':[1]*10000+[42]})
    full,_=grouped_analysis(large,'g',['v'])
    assert full['table']['data'][-1]==['B',42.]


def test_invalid_lab_requests():
    df=fixture()
    for kwargs in [{'group':'absent'},{'group':'species','columns':['species']},
                   {'group':'species','columns':[]},{'group':'species','columns':['sepal_length'],'statistic':'bad'},
                   {'group':'species','columns':['sepal_length'],'layout':'bad'}]:
        with pytest.raises(ValueError):grouped_analysis(df,**kwargs)
    with pytest.raises(ValueError,match='30 groups'):
        grouped_analysis(pd.DataFrame({'g':range(31)}),'g',statistic='count')


def test_bridge_lab_exports_and_stale_invalidation(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from PySide6.QtWidgets import QFileDialog
    from mlstudio.data import DatasetVersions,Transformation
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(fixture())
    c.refresh_data()
    b.command('lab_analyze',json.dumps({'group':'species','columns':['sepal_length']}))
    assert b.snapshot()['lab']['table']['data'][0]==['A',3.]
    assert b.lab_result['chart'].startswith('data:image/svg+xml;base64,')
    for suffix in ['csv','svg','png']:
        output=tmp_path/('summary.'+suffix)
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args,p=output:(str(p),''))
        b.command('lab_export',json.dumps({'format':suffix}))
        assert output.exists() and output.stat().st_size>30
        if suffix=='csv':assert pd.read_csv(output)['sepal_length'].tolist()==[3.,7.,11.]
    c.dataset.apply(Transformation('Remove duplicates',[]))
    assert b.snapshot()['lab'] is None
    c.close()
