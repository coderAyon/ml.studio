"""Embedded canvas acceptance: drag/drop, mouse connections, auto execution and output inspection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings,QPoint,Qt,QEventLoop,QTimer
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .ui_smoke import evaluate,wait_for,capture


def run(output=None):
    out=Path(output or 'examples/reactive-flow-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1880,1100);w.show();js=lambda s:evaluate(w,s)
    def pause(ms=200):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def point(selector):
        p=json.loads(js("JSON.stringify((()=>{const r=document.querySelector("+json.dumps(selector)+").getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})())"))
        return QPoint(round(p['x']),round(p['y']))
    def connect(a,z):
        js("flowUI.inspector=false;render();true");pause()
        surface=w.view.focusProxy() or w.view
        start,end=point('[data-flow-out="'+a+'"]'),point('[data-flow-in="'+z+'"]')
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(surface,end,180);QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
        wait_for(w,"state.workflow.edges.some(e=>e[0]==="+json.dumps(a)+"&&e[1]==="+json.dumps(z)+")&&!state.busy&&!state.flow.stale")
        pause(150)
    def add(kind,x,y):
        wait_for(w,'!state.busy');pause(250)
        count=len(c.canvas.graph.nodes)
        js("(()=>{const canvas=document.querySelector('.wf-canvas'),r=canvas.getBoundingClientRect(),transfer=new DataTransfer();transfer.setData('application/x-mlstudio-node',"+json.dumps(kind)+");canvas.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer,clientX:r.left+"+str(x+52)+",clientY:r.top+"+str(y+52)+"}));return true})()")
        wait_for(w,'Object.keys(state.workflow.nodes).length==='+str(count+1)+'&&!state.busy&&!state.flow.stale');pause(150)
    wait_for(w,"document.body.dataset.ready==='true'")
    notices=[];b.notice.connect(lambda s:notices.append(json.loads(s)))
    old_open,old_save=QFileDialog.getOpenFileName,QFileDialog.getSaveFileName
    try:
        nav=json.loads(js("JSON.stringify(Array.from(document.querySelectorAll('#navigation [data-page]')).map(n=>n.dataset.page))"))
        assert nav[nav.index('data')+1]=='dataset_visuals' and 'analytics' not in nav and 'workflow' in nav
        js("studioUI.navigate('workflow');true")
        assert js("document.querySelectorAll('[data-flow-kind]').length===38")
        add('Dataset',60,100)
        assert b.flow.public['nodes']['1']['status']=='waiting'
        f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'target':[0,1]*40});f.loc[3,'x']=np.nan
        source=out/'synthetic-flow.csv';f.to_csv(source,index=False)
        QFileDialog.getOpenFileName=lambda *a:(str(source),'')
        js("flowSelect('1');true")  # Dropping a step keeps the canvas free; inspect on click.
        js("document.querySelector('[data-flow-source=file]').click();true")
        wait_for(w,'state.dataset?.rows===80&&!state.busy');js("studioUI.navigate('workflow');true")
        wait_for(w,"state.flow.outputs['1']?.status==='ready'&&!state.busy")
        add('Impute missing',290,100);connect('1','2')
        wait_for(w,"state.flow.outputs['2']?.status==='ready'")
        add('Prepared output',520,100);connect('2','3')
        wait_for(w,"state.flow.outputs['3']?.kind==='prepared'&&!state.busy",timeout=60)
        assert b.flow.public['nodes']['3']['training_rows']==60
        assert 'No outcome selected' in b.flow.public['nodes']['3']['note']
        js("flowSelect('3');true")
        assert js("document.querySelector('.wf-output').textContent.includes('Prepared training values')")
        capture(w,out,'workflow');(out/'workflow.png').replace(out/'prepared-output-light.png')
        checks=['Main Visuals navigation and consolidated questions','38 categorized draggable steps','Import node waits until a real CSV is loaded',
                'Native mouse port drag creates arrows and automatically computes','Real training-only prepared output appears on result click']
        export=out/'prepared.csv';QFileDialog.getSaveFileName=lambda *a:(str(export),'')
        js("document.querySelector('[data-flow-export]').click();true")
        assert export.is_file()
        js("flowUI.inspector=false;render();true");pause()
        before=list(c.canvas.graph.nodes['2']['position'])
        surface=w.view.focusProxy() or w.view;start=point('[data-flow-node="2"]')
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(surface,start+QPoint(20,30),160);QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start+QPoint(20,30))
        wait_for(w,"state.workflow.nodes['2'].position[1]>"+str(before[1]+10))
        assert b.flow.public['nodes']['3']['status']=='ready'
        checks.append('Native round-node drag updates arrow geometry without re-fitting; output export works')
        c.target.setCurrentText('target');c.algorithm.setCurrentText('Decision Tree');b.publish()
        wait_for(w,'!state.busy&&!state.flow.stale');pause(150)
        add('Train model',290,330);connect('1','4')
        add('Results',520,330);connect('4','5')
        wait_for(w,"state.flow.outputs['5']?.kind==='model'&&!state.busy",timeout=60)
        assert b.flow.public['nodes']['5']['model']['metadata']['algorithm']=='Decision Tree'
        js("flowSelect('5');true")
        assert js("document.querySelector('.wf-output').textContent.includes('Accuracy')")
        js("cmd('flow_activate',{id:'5'});true")
        wait_for(w,'!!state.model');assert c.result.metadata['target']=='target'
        checks.append('Data branch trains and Results node exposes actual metrics; model activation succeeds')
        project=out/'workflow.json';c.save_project(str(project))
        payload=json.loads(project.read_text());assert len(payload['workflow']['nodes'])==5
        checks.append('Workflow, node settings and connections saved with the project')
        js("flowUI.inspector=false;render();true");capture(w,out,'workflow')
        js("studioUI.navigate('dataset_visuals');true")
        wait_for(w,"!!document.querySelector('[data-visuals-tab=questions]')")
        js("document.querySelector('[data-visuals-tab=questions]').click();true")
        assert js("!!document.querySelector('#visual-question')")
        js("studioUI.navigate('analytics');true")
        assert js("view.page==='dataset_visuals'&&visualsHub.tab==='questions'")
        checks.append('Charts and existing question editor live under Visuals; old links redirect')
        js("studioUI.navigate('workflow');studioUI.command('preferences',{theme:'dark'});true")
        wait_for(w,"document.documentElement.dataset.theme==='dark'");capture(w,out,'workflow');(out/'workflow.png').replace(out/'workflow-dark.png')
        pd.testing.assert_frame_equal(c.dataset.current,f)
        assert not [n for n in notices if n['kind']=='error'],notices
        import sys
        result={'passed':True,'frozen':bool(getattr(sys,'frozen',False)),'checks':checks}
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    finally:
        QFileDialog.getOpenFileName,QFileDialog.getSaveFileName=old_open,old_save
        if not c.busy:w.close()
        app.processEvents()


if __name__=='__main__':run()
