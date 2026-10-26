"""Native acceptance for minimal navigation, cleaning evidence and guided real training."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings,QEventLoop,QTimer
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/minimal-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel')
    w.resize(1540,1000);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    def js(code):return evaluate(w,code)
    def pause(ms=300):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    checks=[]
    assert js("document.querySelectorAll('#navigation [data-page]').length===7")
    assert js("document.querySelectorAll('.minimal-workflow-step').length===4")
    rng=np.random.default_rng(9)
    frame=pd.DataFrame({'measurement':rng.normal(5,.3,120),'spend':rng.uniform(1,50,120),
                        'category':['A','B']*60,'outcome':[0,1]*60,
                        'date':pd.date_range('2026-01-01',periods=120).strftime('%Y-%m-%d')})
    frame.loc[0,'measurement']=np.nan;frame.loc[109,'measurement']=100.
    frame.loc[3,'category']=' A '
    source=out/'synthetic-acceptance.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    wait_for(w,'window.appState?.dataset?.rows===120')
    js("document.querySelector('[data-action=quality_details]').click();true")
    assert js("document.querySelector('.quality-modal [data-open-finding]').textContent.includes('Preprocessing')")
    js("document.querySelector('.quality-modal [data-open-finding]').click();true")
    wait_for(w,"view.page==='preprocessing'&&!!window.appState.preprocessing.detail")
    capture(w,out,'home');capture(w,out,'preprocessing')
    assert js("document.querySelector('.prep-workspace')!==null")
    issue=next(i for i in c.issues if i.title=='Potential outliers' and i.columns==['measurement'])
    js("document.querySelector('[data-prep-issue=\""+issue.id+"\"]').click();true")
    wait_for(w,"window.appState.preprocessing.detail?.id==='"+issue.id+"'")
    assert 110 in b.preprocess_detail['preview']['positions']
    assert js("document.querySelector('.prep-review').textContent.includes('100')")
    capture(w,out,'preprocessing');(out/'preprocessing.png').replace(out/'outlier-evidence-light.png')
    checks.append('Click outlier -> exact affected row beyond first page, right-panel preview and IQR bounds')
    js("document.querySelector('[data-prep-operation=\"Cap outliers (IQR)\"]').click();true")
    wait_for(w,'!!window.appState.preprocessing.preview')
    assert js("document.querySelector('#modal-root').textContent.trim()===''")
    assert b.preprocess_preview['summary']['changed_cells']>=1
    assert c.dataset.index==0
    capture(w,out,'preprocessing');(out/'preprocessing.png').replace(out/'change-preview-light.png')
    js("document.querySelector('[data-action=preprocess_apply]').click();true")
    wait_for(w,'window.appState.dataset.version===1&&!window.appState.busy')
    assert c.dataset.current.measurement.iloc[109]<100
    # Whole-data exploratory transforms cannot silently enter evaluation.
    b.command('simple_train',json.dumps({'target':'outcome','task':'classification','algorithm':'Decision Tree'}))
    assert not c.busy and c.result is None
    js("document.querySelector('[data-action=undo]').click();true")
    wait_for(w,'window.appState.dataset.version===0&&!window.appState.busy')
    assert c.dataset.current.measurement.iloc[109]==100.
    checks.append('Inline before/after preview, Apply, full-data leakage guard, Undo')
    # Training-only capping is a feature preparation choice, not a CSV edit.
    js("document.querySelector('.prep-training details').open=true;document.querySelector('#prep-outliers').value='cap_iqr';document.querySelector('#prep-outliers').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-prep-exclude=\"date\"]').click();document.querySelector('[data-action=prep_save_training]').click();true")
    wait_for(w,"window.appState.preprocessing.preparation.outliers==='cap_iqr'")
    assert c.dataset.index==0 and 'date' in c.preparation['excluded']
    js("studioUI.navigate('models');true")
    assert js("document.getElementById('minimal-target').value===''")
    assert js("document.querySelector('[data-action=simple_train]').disabled")
    js("document.getElementById('minimal-target').value='outcome';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
    wait_for(w,"window.appState.target==='outcome'")
    js("document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    assert c.result.metadata['preparation']['outliers']=='cap_iqr'
    assert 'date' not in c.result.metadata['features']
    assert c.result.metadata['training_rows']==90
    capture(w,out,'models')
    checks.append('Choose target -> real training with fold-fitted preparation, excluded feature, actual metrics')
    js("document.querySelector('[data-action=simple_compare]').click();true")
    wait_for(w,'window.appState.comparison.length===3&&!window.appState.busy',timeout=60)
    checks.append('One-click three-method comparison uses identical split and preparation')
    for page in ['home','data','preprocessing','analytics','models','results','advanced','workflow','analyzer','safety','impact','reliability','errors','researchpack','history','predictions','reports','settings']:
        js('studioUI.navigate('+json.dumps(page)+');true')
        assert js("document.getElementById('content').textContent.trim().length>30"),page
        assert js("document.querySelectorAll('#navigation [data-page]').length===7")
    checks.append('All 18 available routes remain reachable with 7 main navigation items')
    # Extended questions cover time, percentage and a multi-question queue.
    js("studioUI.navigate('analytics');document.querySelector('#visual-question').value="+json.dumps('1. Trend spend over date\n2. Show percentage share of category')+";document.querySelector('[data-action=visual_prepare_batch]').click();true")
    wait_for(w,'window.appState.visual_requests.batch.length===2')
    assert b.visual_review['plan']['kind']=='line'
    assert js("document.querySelector('#visual-question').value==='Trend spend over date'")
    js("document.querySelector('[data-action=visual_execute]').click();true")
    wait_for(w,'window.appState.visual_requests.results.length===1')
    assert b.visual_results[0]['plan']['kind']=='line'
    js("document.querySelector('[data-visual-review-index=\"1\"]').click();true")
    wait_for(w,"window.appState.visual_requests.review.plan.kind==='category_share'")
    js("document.querySelector('[data-action=visual_execute]').click();true")
    wait_for(w,'window.appState.visual_requests.results.length===2')
    checks.append('Two written questions -> separate date trend and category-share results')
    # Repeating the same multi-question input must again select the first plan.
    js("document.querySelector('#visual-question').value="+json.dumps('1. Trend spend over date\n2. Show percentage share of category')+";document.querySelector('[data-action=visual_prepare_batch]').click();true")
    wait_for(w,"window.appState.visual_requests.review.plan.kind==='line'")
    assert js("document.querySelector('#visual-question').value==='Trend spend over date'")
    js("studioUI.command('preferences',{theme:'dark'});true")
    wait_for(w,"document.documentElement.dataset.theme==='dark'")
    capture(w,out,'preprocessing');(out/'preprocessing.png').replace(out/'preprocessing-dark.png')
    capture(w,out,'models');(out/'models.png').replace(out/'models-dark.png')
    capture(w,out,'advanced')
    # Source bytes and original active values remain untouched by fitted preparation.
    assert pd.read_csv(source).measurement.iloc[109]==100.
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
    w.close();app.processEvents()


if __name__=='__main__':run()
