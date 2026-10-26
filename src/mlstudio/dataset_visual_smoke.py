"""Native acceptance of the Dataset visuals destination and its chart controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/dataset-visuals-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel');w.resize(1530,980);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    frame=pd.DataFrame({'length':[1.,2.,3.,4.]*30+[100.],
                        'width':[2.,4.,6.,8.]*30+[200.],'species':['A','B','C']*40+['A']})
    source=out/'synthetic-visuals.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    def js(code):return evaluate(w,code)
    wait_for(w,'window.appState?.dataset?.rows===121')
    order=json.loads(js("JSON.stringify(Array.from(document.querySelectorAll('#navigation [data-page]')).map(el=>el.dataset.page))"))
    assert order[order.index('data')+1]=='dataset_visuals'
    js("document.querySelector('#navigation [data-page=dataset_visuals]').click();true")
    wait_for(w,"window.appState.dataset_visuals?.kind==='scatter'")
    assert js("document.querySelectorAll('[data-dv-kind]').length===5")
    assert b.dataset_visual_result['outliers']['count']==1
    js("document.querySelector('[data-action=dv_rows]').click();true")
    assert js("document.querySelector('.dv-outlier-rows').open")
    assert js("document.querySelector('.dv-outlier-rows .table-scroll').textContent.includes('121')")
    capture(w,out,'dataset_visuals');(out/'dataset_visuals.png').replace(out/'scatter-light.png')
    checks=['Dataset visuals immediately below My dataset; automatic scatter and exact flagged row 121']
    for kind in ['histogram','box','bar','correlation']:
        js("document.querySelector('[data-dv-kind="+kind+"]').click();true")
        wait_for(w,"window.appState.dataset_visuals.kind==='"+kind+"'")
        assert js("document.querySelector('.dv-plot img').naturalWidth>0")
    checks.append('All five chart tabs produce real dataset charts')
    for method in ['spearman','kendall','pearson']:
        js("document.querySelector('#dv-correlation').value='"+method+"';document.querySelector('#dv-correlation').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-action=dv_draw]').click();true")
        wait_for(w,"window.appState.dataset_visuals.config.correlation==='"+method+"'")
        assert b.dataset_visual_result['pairs']['data'][0][2]>.99
        assert b.dataset_visual_result['pairs']['data'][0][3]==121
    checks.append('Pearson, Spearman and Kendall share actual pair-count evidence')
    js("document.querySelector('.dv-customize').open=true;document.querySelector('#dv-title').value='My relationship map';document.querySelector('#dv-title').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-action=dv_draw]').click();true")
    wait_for(w,"window.appState.dataset_visuals.title==='My relationship map'")
    js("document.querySelector('[data-action=dv_expand]').click();true")
    assert js("document.body.classList.contains('dv-chart-expanded')")
    js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));true")
    assert js("!document.body.classList.contains('dv-chart-expanded')")
    capture(w,out,'dataset_visuals');(out/'dataset_visuals.png').replace(out/'correlation-light.png')
    old=QFileDialog.getSaveFileName
    try:
        for fmt in ['png','svg','csv','outliers','correlation_pairs']:
            path=out/('export-'+fmt+('.csv' if fmt in {'outliers','correlation_pairs'} else '.'+fmt))
            QFileDialog.getSaveFileName=lambda *a,p=path:(str(p),'')
            b.command('dataset_visual_export',json.dumps({'format':fmt}))
            assert path.exists() and path.stat().st_size>30
    finally:QFileDialog.getSaveFileName=old
    checks.append('Custom title, expand/Escape and actual image/table/outlier/pair exports')
    js("studioUI.command('preferences',{theme:'dark'});true")
    wait_for(w,"document.documentElement.dataset.theme==='dark'")
    capture(w,out,'dataset_visuals');(out/'dataset_visuals.png').replace(out/'correlation-dark.png')
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
    w.close();app.processEvents()


if __name__=='__main__':run()
