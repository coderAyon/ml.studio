"""Observable request parsing, real chart calculations and bridge exports."""
import json
import base64
import numpy as np
import pandas as pd
import pytest
from mlstudio.visual_requests import parse_request,parse_requests,execute_request
from test_lab import fixture


@pytest.mark.parametrize('question,statistic,layout,columns',[
    ('Create a graph that shows the avg value of sepal_length for each species','mean','combined',['sepal_length']),
    ('Count the number of flowers belonging to each species and visualize using a bar chart','count','combined',[]),
    ('Calculate average values of the four numerical features for each species using three vertically arranged subplots','mean','subplots',['sepal_length','sepal_width','petal_length','petal_width']),
    ('প্রতি species এর গড় sepal_length দেখাও','mean','combined',['sepal_length']),
    ('species er count dekhao','count','combined',[]),
])
def test_lab_requests_and_language_matching(question,statistic,layout,columns):
    result=parse_request(fixture(),question)
    assert result['ready'],result['questions']
    p=result['plan']
    assert p['group']=='species' and p['statistic']==statistic
    assert p['layout']==layout and p['columns']==columns
    output,figure=execute_request(fixture(),p)
    assert len(output['table']['data'])==3
    assert output['rows_used']==6


def test_ambiguous_unsupported_and_missing_column_requests():
    df=fixture().assign(region=['N','S']*3)
    for question in ['Show average salary by department','Pie chart of species count',
                     'Histogram and scatter of sepal_length and sepal_width',
                     'Mean and median sepal_length by species',
                     'Average sepal_length by species where sepal_width > 3']:
        result=parse_request(df,question)
        assert result['questions'] and not result['ready'],question
    assert not parse_request(df,'show avg sepal_length')['plan']['group']
    with pytest.raises(ValueError):parse_request(df,'x'*4001)


def test_column_aliases_and_scatter_mention_order():
    df=fixture().rename(columns={'sepal_length':'Sepal Length (cm)','sepal_width':'Sepal Width'})
    p=parse_request(df,'Scatter of Sepal Width versus Sepal Length')['plan']
    assert p['columns']==['Sepal Width','Sepal Length (cm)']
    # Numeric group names are allowed when they follow an explicit 'by'.
    p=parse_request(fixture(),'Average sepal_length by petal_width')['plan']
    assert p['group']=='petal_width' and p['columns']==['sepal_length']


def test_top_n_sort_horizontal_and_filter_use_full_correct_rows():
    df=fixture();before=df.copy(deep=True)
    p=parse_request(df,'Top 2 species by average sepal_length horizontal bar chart')['plan']
    result,fig=execute_request(df,p)
    assert result['table']['data']==[['C',11.],['B',7.]]
    assert [bar.get_width() for bar in fig.axes[0].patches]==[11.,7.]
    p.update(top_n=None,filter_column='sepal_width',filter_operator='gt',filter_value='6')
    result,_=execute_request(df,p)
    assert result['rows_used']==3 and result['table']['data']==[['C',11.],['B',8.]]
    pd.testing.assert_frame_equal(df,before)
    p['filter_value']='100'
    with pytest.raises(ValueError,match='No rows'):execute_request(df,p)
    p['filter_value']='bad'
    with pytest.raises(ValueError,match='numeric'):execute_request(df,p)


def test_all_chart_types_have_real_values_and_missing_rules():
    df=fixture();df.loc[0,'sepal_length']=np.inf;df.loc[1,'sepal_width']=np.nan
    for question in ['Histogram sepal_length with 3 bins',
                     'Scatter sepal_length versus sepal_width',
                     'Box plot sepal_length and petal_width',
                     'Correlation sepal_length and sepal_width',
                     'Show missing values']:
        parsed=parse_request(df,question)
        assert parsed['ready'],parsed['questions']
        result,fig=execute_request(df,parsed['plan'])
        assert result['table']['data'] and result['protocol'] and result['insights']
        json.dumps(result,allow_nan=False)
        if parsed['plan']['kind']=='histogram':assert sum(row[2] for row in result['table']['data'])==5
        if parsed['plan']['kind']=='scatter':assert result['table']['data'][0][1]==4
        if parsed['plan']['kind']=='missing':assert sum(row[1] for row in result['table']['data'])==1


def test_large_scatter_sampling_does_not_limit_correlation():
    df=pd.DataFrame({'x':range(6000),'y':range(6000)})
    result,_=execute_request(df,parse_request(df,'scatter x versus y')['plan'])
    assert result['table']['data'][0][1]==6000
    assert result['table']['data'][2][1]==5000
    assert result['table']['data'][1][1]==pytest.approx(1.)


def test_bridge_gallery_exports_theme_and_version(tmp_path,monkeypatch):
    from test_bridge import controller_with_data
    from mlstudio.data import DatasetVersions,Transformation
    from PySide6.QtWidgets import QFileDialog
    app,c,b=controller_with_data(tmp_path)
    c.dataset=DatasetVersions(fixture());c.refresh_data();b.snapshot()
    b.command('visual_prepare',json.dumps({'question':'average sepal_length by species'}))
    p=b.visual_review['plan']
    for i in range(7):b.command('visual_execute',json.dumps({'question':'<script>bad</script>','plan':p}))
    assert len(b.visual_results)==6 and len(b.visual_figures)==6
    r=b.visual_results[0]
    assert r['table']['data'][0]==['A',3.]
    for format in ['csv','png','svg','html','json']:
        dest=tmp_path/('visual.'+format)
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,p=dest:(str(p),''))
        b.command('visual_export',json.dumps({'id':r['id'],'format':format}))
        assert dest.exists() and dest.stat().st_size>30
    html=(tmp_path/'visual.html').read_text(encoding='utf-8')
    assert '&lt;script&gt;bad&lt;/script&gt;' in html and '<script>bad</script>' not in html
    assert json.loads((tmp_path/'visual.json').read_text(encoding='utf-8'))['plan']==p
    b.command('preferences','{"theme":"dark"}')
    assert '#1a2133' in base64.b64decode(r['chart'].split(',',1)[1]).decode()
    b.command('visual_remove',json.dumps({'id':r['id']}))
    assert len(b.visual_results)==len(b.visual_figures)==5
    c.dataset.apply(Transformation('Remove duplicates',[]))
    cleared=b.snapshot()['visual_requests']
    assert cleared['review'] is None and cleared['results']==[] and not cleared.get('batch')
    assert not b.visual_figures
    c.close()


def test_trend_sorts_dates_aggregates_repeated_x_and_keeps_series_separate():
    df=pd.DataFrame({'date':['2025-01-03','2025-01-01','2025-01-01','2025-01-02','2025-01-01'],
                     'sales':[90.,10.,30.,50.,100.],'region':['N','N','N','N','S']})
    review=parse_request(df,'Trend sales over date by region')
    assert review['ready'],review['questions']
    result,fig=execute_request(df,review['plan'])
    assert result['table']['data']==[['N','2025-01-01T00:00:00',20.],
                                     ['N','2025-01-02T00:00:00',50.],
                                     ['N','2025-01-03T00:00:00',90.],
                                     ['S','2025-01-01T00:00:00',100.]]
    assert len(fig.axes[0].lines)==2
    assert fig.axes[0].lines[0].get_ydata().tolist()==[20.,50.,90.]
    assert fig.axes[0].lines[1].get_ydata().tolist()==[100.]
    assert result['support']['data'][0][-1]==2
    assert 'not a forecast' in result['insights'][-1]
    json.dumps(result,allow_nan=False)


def test_numeric_line_axis_requires_explicit_selection_and_valid_dates():
    df=pd.DataFrame({'week':[3,1,2,2],'sales':[30.,10.,np.inf,20.]})
    review=parse_request(df,'Line chart sales over week')
    assert review['ready'] and review['plan']['x_column']=='week'
    result,fig=execute_request(df,review['plan'])
    assert result['table']['data']==[[1.,10.],[2.,20.],[3.,30.]]
    assert result['support']['data'][1][-1]==1
    ambiguous=parse_request(df,'Trend sales and week')
    assert not ambiguous['ready'] and not ambiguous['plan']['x_column']
    bad=pd.DataFrame({'date':['01/02/2025','03/02/2025'],'sales':[1.,2.]})
    p=parse_request(bad,'Trend sales over date')['plan']
    with pytest.raises(ValueError,match='YYYY-MM-DD'):execute_request(bad,p)
    invalid=pd.DataFrame({'date':['2025-13-01','2025-01-02'],'sales':[1.,2.]})
    with pytest.raises(ValueError,match='invalid'):execute_request(invalid,p)


def test_line_series_and_point_limits_and_missing_values():
    df=pd.DataFrame({'x':[1.,2.,3.],'value':[1.,np.nan,3.]})
    p=parse_request(df,'Trend value over x')['plan']
    result,fig=execute_request(df,p)
    assert result['table']['data'][1]==[2.,None]
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[1])
    many=pd.DataFrame({'x':range(5001),'value':range(5001)})
    with pytest.raises(ValueError,match='5,000'):execute_request(many,p)
    many_series=pd.DataFrame({'x':[1]*13,'value':range(13),'group':list('abcdefghijklm')})
    p['series_column']='group'
    with pytest.raises(ValueError,match='12'):execute_request(many_series,p)


def test_category_percentages_use_filtered_denominator_without_topn_renormalizing():
    df=pd.DataFrame({'category':['A','A','A','B',None],'value':[1,2,3,4,5]})
    p=parse_request(df,'Top 1 percentage share of category')['plan']
    assert p['kind']=='category_share' and p['group']=='category'
    result,fig=execute_request(df,p)
    assert result['table']['data']==[['A',3,60.]]
    assert fig.axes[0].patches[0].get_height()==60.
    assert 'not renormalize' in result['protocol']
    p.update(top_n=None,filter_column='value',filter_operator='gt',filter_value=2)
    result,_=execute_request(df,p)
    assert result['rows_used']==3 and len(result['table']['data'])==3
    assert sum(row[2] for row in result['table']['data'])==pytest.approx(100.)
    assert any(row[0]=='(missing group)' for row in result['table']['data'])


def test_numbered_question_batch_prepares_without_executing():
    reviews=parse_requests(fixture(),'1. Average sepal_length by species\n2. Histogram petal_width\n3. Percentage share of species')
    assert len(reviews)==3 and all(r['ready'] for r in reviews)
    assert [r['plan']['kind'] for r in reviews]==['grouped','histogram','category_share']
    assert not any('table' in r for r in reviews)
    wrapped=parse_requests(fixture(),'Problem1: Average sepal_length\nby species\nProblem2: Histogram petal_width')
    assert len(wrapped)==2 and wrapped[0]['plan']['group']=='species'
    with pytest.raises(ValueError,match='six'):parse_requests(fixture(),'\n'.join('Histogram sepal_length' for _ in range(7)))
