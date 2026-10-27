"""Acceptance of explicit import sources and the expanded workflow library."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/expanded-flow-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    w.resize(1740,1060);w.show();js=lambda s:evaluate(w,s);old=QFileDialog.getOpenFileName
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        f=pd.DataFrame({'x':np.arange(120,dtype=float),'z':np.sin(np.arange(120)),'label':[0,1]*60})
        c.dataset=DatasetVersions(f);c.source=str(out/'workspace.csv');f.to_csv(c.source,index=False);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('label');b.publish()
        wait_for(w,'state.dataset?.rows===120');js("studioUI.navigate('workflow');flowAdd('Dataset',70,80);true")
        wait_for(w,"state.flow.outputs['1']?.status==='waiting'&&!state.busy")
        js("flowSelect('1');true")
        assert js("document.querySelector('.wf-output').textContent.includes('No CSV chosen')&&!document.querySelector('.wf-output table')")
        assert js("document.querySelectorAll('[data-flow-kind]').length===38")
        QTest.qWait(300);assert js("Math.round(document.querySelector('.wf-node').getBoundingClientRect().width)===56")
        js("document.querySelector('#toast').classList.remove('show');true");QTest.qWait(250)
        w.grab().save(str(out/'empty-import-node.png'))
        second=out/'independent.csv';f.iloc[:40].to_csv(second,index=False);QFileDialog.getOpenFileName=lambda *a:(str(second),'')
        js("document.querySelector('[data-flow-source=file]').click();true")
        wait_for(w,"state.flow.outputs['1']?.rows===40&&!state.busy&&!state.flow.stale",timeout=60)
        assert len(c.dataset.current)==120
        js("flowUI.inspector=false;flowAdd('Dataset',70,260);true")
        wait_for(w,"state.flow.outputs['2']?.status==='waiting'&&!state.busy")
        js("cmd('flow_use_workspace',{id:'2'});true")
        wait_for(w,"state.flow.outputs['2']?.rows===120&&!state.busy&&!state.flow.stale")
        assert b.flow.public['nodes']['1']['rows']==40
        js("flowAdd('Statistics',280,80);true");wait_for(w,'!state.busy&&!state.flow.stale')
        js("flowConnect('1','3');true");wait_for(w,"state.flow.outputs['3']?.kind==='analysis'&&!state.busy&&!state.flow.stale")
        js("flowSelect('3');true");assert js("document.querySelector('.wf-output').textContent.includes('Mean')")
        js("flowUI.inspector=false;flowAdd('K-Means',290,270);true");wait_for(w,'!state.busy&&!state.flow.stale')
        js("flowConnect('2','4');true");wait_for(w,"state.flow.outputs['4']?.kind==='analysis'&&!state.busy&&!state.flow.stale",timeout=60)
        assert 'Cluster' in b.flow.engine.values['4'].frame
        js("flowUI.inspector=false;flowUI.collapsed.add('Learn');render();const s=document.getElementById('wf-search');s.value='Random Forest';s.dispatchEvent(new Event('input',{bubbles:true}));true")
        assert js("!document.querySelector('[data-flow-group=Learn] .wf-library-items').hidden")
        js("flowUI.search='';render();flowSelect('4');document.querySelector('#toast').classList.remove('show');true");QTest.qWait(500)
        w.grab().save(str(out/'expanded-library-clustering.png'))
        (out/'result.json').write_text(json.dumps({'passed':True,'checks':['Import node is empty despite loaded workspace',
            'Choose CSV is visible without opening settings','38 real catalog entries','56px nodes',
            'Per-node file import keeps existing workspace unchanged','Two independent data sources retain different row counts',
            'Column statistics show real values','Clustering yields cluster labels','Library search expands collapsed matches']},indent=2))
        print('Expanded workflow desktop smoke passed')
    finally:QFileDialog.getOpenFileName=old;w.close();app.processEvents()


if __name__=='__main__':run()
