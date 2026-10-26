import json
import numpy as np
import pandas as pd
import pytest
from mlstudio.dataset_visuals import build_chart,detect_outliers


def data():
    return pd.DataFrame({'x':[1.,2.,3.,4.]*10+[100.,np.nan],
                         'y':[2.,4.,6.,8.]*10+[200.,12.],
                         'group':['A','B']*21})


def test_outlier_rules_count_exact_positions_and_exclude_missing_infinity():
    frame=data();frame.loc[0,'y']=np.inf
    for rule in ['iqr','zscore']:
        mask,fields=detect_outliers(frame,['x','y'],rule)
        assert np.flatnonzero(mask).tolist()==[40]
        assert fields[0]['finite_count']==41
        assert fields[1]['infinite_count']==1
    mask,_=detect_outliers(pd.DataFrame({'x':[4.,4.,np.nan]}),['x'],'zscore')
    assert not mask.any()
    with pytest.raises(ValueError):detect_outliers(frame,['group'])


@pytest.mark.parametrize('kind,columns',[('scatter',['x','y']),('histogram',['x']),('box',['x','y']),('bar',[])])
def test_customized_charts_are_real_and_preserve_source(kind,columns):
    frame=data();before=frame.copy(deep=True)
    result,figure=build_chart(frame,{'kind':kind,'columns':columns,'group':'group','title':'My chart',
                                    'xlabel':'Horizontal','ylabel':'Vertical','color':'teal','bins':8})
    assert figure.axes[0].get_title()=='My chart'
    assert figure.axes[0].get_xlabel()=='Horizontal'
    assert result['table']['data']
    json.dumps(result,allow_nan=False)
    if kind=='histogram':assert sum(row[2] for row in result['table']['data'])==41
    if kind=='bar':assert result['table']['data']==[['A',21.],['B',21.]]
    if kind=='scatter':assert result['outliers']['preview']['data'][0]==[41,100.,200.]
    pd.testing.assert_frame_equal(frame,before)


@pytest.mark.parametrize('method',['pearson','spearman','kendall'])
def test_correlation_variants_report_coefficients_and_pair_counts(method):
    frame=pd.DataFrame({'x':[1.,2.,3.,4.,np.nan],'y':[2.,4.,6.,8.,10.], 'constant':[1.]*5})
    result,figure=build_chart(frame,{'kind':'correlation','columns':['x','y','constant'],'correlation':method})
    assert result['table']['data'][0][2]==pytest.approx(1.)
    assert result['support']['data'][0][2]==4
    assert result['pairs']['data'][0]==['x','y',pytest.approx(1.),4]
    assert result['table']['data'][0][3] is None
    json.dumps(result,allow_nan=False)


def test_bridge_exports_flagged_rows_and_invalidates_on_version_change(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from mlstudio.data import DatasetVersions,Transformation
    from PySide6.QtWidgets import QFileDialog
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(data());c.refresh_data();b.snapshot()
    b.command('dataset_visual_chart',json.dumps({'kind':'scatter','columns':['x','y']}))
    assert b.snapshot()['dataset_visuals']['outliers']['count']==1
    assert b.dataset_visual_result['chart'].startswith('data:image/svg+xml;base64,')
    for fmt in ['png','svg','csv','outliers']:
        path=tmp_path/('export-'+fmt+('.csv' if fmt=='outliers' else '.'+fmt))
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,p=path:(str(p),''))
        b.command('dataset_visual_export',json.dumps({'format':fmt}))
        assert path.exists() and path.stat().st_size>30
        if fmt=='outliers':assert pd.read_csv(path)['active_row'].tolist()==[41]
    b.command('dataset_visual_chart',json.dumps({'kind':'correlation','columns':['x','y'],'correlation':'spearman'}))
    path=tmp_path/'pairs.csv';monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a:(str(path),''))
    b.command('dataset_visual_export','{"format":"correlation_pairs"}')
    assert len(pd.read_csv(path))==1
    c.dataset.apply(Transformation('Remove duplicates',[]))
    assert b.snapshot()['dataset_visuals'] is None
    c.close()


def test_invalid_customization_rejected():
    for cfg in [{'kind':'scatter','columns':['x','x']},{'kind':'histogram','columns':['x'],'bins':1},
                {'kind':'correlation','columns':['x','y'],'correlation':'magic'},
                {'kind':'scatter','columns':['x','y'],'opacity':2}]:
        with pytest.raises(ValueError):build_chart(data(),cfg)
