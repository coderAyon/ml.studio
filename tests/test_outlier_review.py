import numpy as np
import pandas as pd
from mlstudio.analyzer import profile
from mlstudio.data import DatasetVersions,Transformation
from mlstudio.dataset_visuals import build_chart


def fixture():
    f=pd.DataFrame({'length':np.linspace(1,5,150),'width':np.linspace(2,4,150),'label':['a','b']*75})
    f.loc[58,'length']=200;f.loc[140,'length']=300
    f.index=['same']*len(f)
    return f


def test_column_information_reports_exact_outliers_and_extreme_examples():
    result=profile(fixture());field=result['fields'][0]
    assert field['outliers']['outlier_count']==2
    assert field['outliers']['upper']<200
    assert {'row':59,'value':200.0} in field['outlier_examples']


def test_table_markers_follow_positions_across_sort_filter_and_pages(tmp_path):
    from test_bridge import controller_with_data
    app,c,b=controller_with_data(tmp_path);c.dataset=DatasetVersions(fixture());c.source='synthetic.csv'
    result=b.dataset_state()
    assert result['outlier_cells']==2 and result['outlier_rows']==2
    assert result['outlier_marks']==[{'row':59,'column':'length'}]
    b.table_options={'page':1};result=b.dataset_state()
    assert result['outlier_marks']==[{'row':141,'column':'length'}]
    b.table_options={'sort':'length','descending':True,'outliers_only':True};result=b.dataset_state()
    assert result['preview']['positions']==[141,59]
    assert result['preview']['data'][1][0]==200
    assert len(result['outlier_marks'])==2
    b.table_options={'outliers_only':True,'search':'200'};result=b.dataset_state()
    assert result['filtered_rows']==1 and result['preview']['positions']==[59]
    b.table_options={'outliers_only':True,'outlier_column':'width'}
    assert b.dataset_state()['preview']['data']==[]
    b.table_options={};c.dataset.apply(Transformation('Filter rows',['label'],'a'))
    result=b.dataset_state();assert result['outlier_rows']==2
    assert {m['row'] for m in result['outlier_marks']}=={30,71}


def test_boxplot_and_scatter_label_extreme_values():
    f=fixture()
    result,fig=build_chart(f,{'kind':'box','columns':['length']})
    assert result['outliers']['count']==2
    assert '200' in [t.get_text() for t in fig.axes[0].texts]
    result,fig=build_chart(f,{'kind':'scatter','columns':['width','length']})
    assert any('200' in t.get_text() and 'Row 59' in t.get_text() for t in fig.axes[0].texts)


def test_scatter_keeps_extreme_points_when_sampling():
    f=pd.DataFrame({'x':np.linspace(1,5,6000),'y':np.linspace(2,6,6000)});f.loc[5777,'y']=200
    result,fig=build_chart(f,{'kind':'scatter','columns':['x','y']})
    assert any('Row 5778' in t.get_text() for t in fig.axes[0].texts)
