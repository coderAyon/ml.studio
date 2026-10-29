from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from mlstudio.business import analyze_business


def fixture():
    return pd.DataFrame({'sales':[100.,200.,50.,np.nan,0.], 'cost':[60.,120.,30.,20.,0.],
        'region':['East','West','East','West',None], 'product':['A','B','B','A','B'],
        'date':['2026-01-05','2026-01-20','2026-02-01','bad','2026-02-02'],
        'customer':['x','y','x','z','z']})


def config():
    return {'measure':'sales','aggregate':'sum','dimension':'region','date':'date','grain':'month',
        'pivot':'product','cost':'cost','customer':'customer','target':500.,'scenario_percent':10.}


def test_actual_kpis_profit_segments_trends_and_no_mutation():
    f=fixture();before=f.copy(deep=True);r=analyze_business(f,config())
    assert r['metric']['value']==350 and r['metric']['finite_rows']==4
    assert r['finance']['profit']==140 and r['finance']['paired_rows']==4
    assert r['target']['gap']==-150 and r['target']['attainment_percent']==70
    assert r['scenario']['value']==pytest.approx(385)
    assert r['distinct_customers']==3
    assert r['trend']['table']['data']==[['2026-01',300.,2],['2026-02',50.,2]]
    assert r['trend']['invalid_dates']==1 and r['trend']['change_percent']==pytest.approx(-250/3)
    assert r['segments']['shown']==3 and len(r['pivot']['data'])==3
    assert r['filtered_rows']==5 and len(r['detail']['data'])==5
    json.dumps(r,allow_nan=False)
    pd.testing.assert_frame_equal(f,before)


def test_filters_scope_all_results_and_keep_exact_source_positions():
    r=analyze_business(fixture(),{**config(),'filters':[{'column':'region','op':'eq','value':'East'},
        {'column':'sales','op':'ge','value':60}]})
    assert r['filtered_rows']==1 and r['metric']['value']==100
    assert r['detail']['data'][0][0]==1
    assert r['distinct_customers']==1
    assert r['trend']['change_percent'] is None


def test_missing_and_negative_values_do_not_fabricate_business_ratios():
    f=fixture();f['sales']=np.nan;r=analyze_business(f,config())
    assert r['metric']['value'] is None and r['finance']['profit'] is None
    assert r['scenario']['value'] is None and r['segments']['pareto_available'] is False
    f['sales']=[-20.,10.,0.,np.inf,np.nan];r=analyze_business(f,{**config(),'target':0})
    assert r['segments']['pareto_available'] is False and r['target']['attainment_percent'] is None
    assert r['metric']['infinite_rows']==1


@pytest.mark.parametrize('patch',[
    {'aggregate':'eval'}, {'measure':'absent'}, {'scenario_percent':float('inf')},
    {'date':'sales'}, {'filters':[{'column':'sales','op':'code','value':'x'}]},
    {'target':True},{'top_n':0},{'grain':'never'}, {'filters':[{'column':'sales','op':'ge','value':'text'}]},
])
def test_invalid_business_config_fails_without_changes(patch):
    f=fixture();before=f.copy(deep=True)
    with pytest.raises(ValueError):analyze_business(f,{**config(),**patch})
    pd.testing.assert_frame_equal(f,before)


def test_count_distinct_null_groups_and_safe_zero_baseline():
    r=analyze_business(fixture(),{'aggregate':'count','dimension':'region','date':'date'})
    assert r['metric']['value']==5
    assert sum(row[1] for row in r['segments']['table']['data'])==5
    r=analyze_business(fixture(),{'aggregate':'distinct','measure':'customer'})
    assert r['metric']['value']==3
    assert r['segments'] is None


def test_missing_segment_label_is_distinct_and_trend_preserves_calendar_gaps():
    f=pd.DataFrame({'group':['(missing)',None,'B'],'value':[1.,2.,3.],
                    'date':['2026-01-01','2026-03-01','2026-04-01']})
    r=analyze_business(f,{'aggregate':'sum','measure':'value','dimension':'group','date':'date'})
    assert r['segments']['missing_label']=='_(missing)'
    p=r['trend']['positions'];assert p[1]-p[0]==2 and p[2]-p[1]==1


def test_business_sum_does_not_wrap_fixed_width_unsigned_integers():
    f=pd.DataFrame({'amount':pd.Series([2**64-1,2**64-1],dtype='uint64')})
    r=analyze_business(f,{'aggregate':'sum','measure':'amount'})
    assert r['metric']['value']==pytest.approx(2.*(2**64-1)) and r['metric']['value']>0


def test_business_bridge_mode_exports_and_revision_safety(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QFileDialog
    from mlstudio.data import DatasetVersions,Transformation
    app,c,b=controller_with_data(tmp_path)
    c.settings=QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat)
    c.dataset=DatasetVersions(fixture());c.source=str(tmp_path/'source.csv')
    c.dataset.current.to_csv(c.source,index=False);original=Path(c.source).read_bytes()
    b.research._background=lambda work,message,complete:complete(work())
    tools=b.business
    assert not tools.state()['enabled']
    tools.command('business_mode',{'enabled':True});assert tools.state()['enabled']
    tools.command('business_run',{'config':config()});assert tools.state()['result']['metric']['value']==350
    for kind in ['html','summary','segments','trend','pivot','data']:
        path=tmp_path/('report-'+kind+('.html' if kind=='html' else '.csv'))
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args,p=path:(str(p),''))
        tools.command('business_export',{'kind':kind});assert path.stat().st_size>10
        if kind!='html':assert b'\r\r\n' not in path.read_bytes()
    definition=tmp_path/'definition.json'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(definition),''))
    tools.command('business_definition_save',{})
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(definition),''))
    tools.command('business_definition_open',{})
    assert c.business_config['measure']=='sales'
    c.dataset.apply(Transformation('Remove duplicates',[]))
    assert tools.state()['result'] is None
    with pytest.raises(ValueError,match='current dataset'):tools.command('business_export',{'kind':'data'})
    tools.command('business_mode',{'enabled':False});assert not tools.state()['enabled']
    assert Path(c.source).read_bytes()==original
    c.close()
