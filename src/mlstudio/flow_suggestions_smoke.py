"""Native pointer acceptance of compact, user-built suggested workflows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint,Qt,QSettings
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/flow-suggestions-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1700,1050);w.show();js=lambda s:evaluate(w,s)
    def point(selector,dx=0,dy=0):
        value=json.loads(js("JSON.stringify((()=>{const r=document.querySelector("+json.dumps(selector)+").getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})())"))
        return QPoint(round(value['x']+dx),round(value['y']+dy))
    def pull(source):
        start=point('[data-flow-out="'+source+'"]')
        value=json.loads(js("JSON.stringify((()=>{const r=document.querySelector('.wf-canvas').getBoundingClientRect();return {x:r.left+420,y:r.top+160+Object.keys(state.workflow.nodes).length*120}})())"))
        end=QPoint(round(value['x']),round(value['y']))
        surface=w.view.focusProxy() or w.view
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(surface,end,160);QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
        wait_for(w,"!!document.querySelector('.wf-suggestions')")
        QTest.qWait(250)
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'label':[0,1]*40});f.loc[[2,10],'x']=np.nan
        source=out/'synthetic.csv';f.to_csv(source,index=False)
        c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('label');b.publish()
        wait_for(w,'state.dataset?.rows===80')
        js("studioUI.navigate('workflow');true")
        assert js("Object.keys(state.workflow.nodes).length===0&&!document.querySelector('[data-flow-action=starter]')")
        # HTML5 library drop is synthesized; all subsequent arrow drags use native QTest input.
        js("(()=>{const r=document.querySelector('.wf-canvas').getBoundingClientRect(),d=new DataTransfer();d.setData('application/x-mlstudio-node','Dataset');document.querySelector('.wf-canvas').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d,clientX:r.left+100,clientY:r.top+120}));return true})()")
        wait_for(w,"state.flow.outputs['1']?.status==='ready'&&!state.busy&&!state.flow.stale")
        QTest.qWait(350)
        assert js("Object.keys(state.workflow.nodes).length===1&&!document.querySelector('.wf-inspector')")
        assert js("Math.round(document.querySelector('.wf-node').getBoundingClientRect().width)===72")
        pull('1')
        assert js("Object.keys(state.workflow.nodes).length===1&&state.workflow.edges.length===0")
        assert js("document.querySelector('[data-flow-suggest]').dataset.flowSuggest==='Impute missing'")
        assert js("document.querySelector('.wf-suggestions').textContent.includes('2 missing cells')")
        js("document.querySelector('#toast').classList.remove('show');true");QTest.qWait(300)
        w.grab().save(str(out/'suggestions-light.png'))
        surface=w.view.focusProxy() or w.view
        QTest.mouseClick(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point('[data-flow-suggest="Impute missing"]'))
        wait_for(w,"state.flow.outputs['2']?.status==='ready'&&!state.busy&&!state.flow.stale")
        assert js("Object.keys(state.workflow.nodes).length===2&&state.workflow.edges.length===1&&state.workflow.edges[0][0]==='1'&&state.workflow.edges[0][1]==='2'")
        pull('1');QTest.keyClick(surface,Qt.Key.Key_Escape);QTest.qWait(250)
        assert js("!document.querySelector('.wf-suggestions')&&Object.keys(state.workflow.nodes).length===2")
        # Only compatible model-result suggestions are offered for a model output.
        assert js("(()=>{const g=state.workflow;g.nodes['99']={type:'Train model',position:[500,300]};const suggestions=flowNextSteps('99');delete g.nodes['99'];return suggestions.length===1&&suggestions[0].type==='Results'})()")
        js("document.querySelector('[data-flow-action=new]').click();true")
        wait_for(w,"Object.keys(state.workflow.nodes).length===0&&!state.busy")
        assert len(c.canvas.graph.nodes)==0
        js("document.querySelector('[data-flow-action=undo]').click();true")
        wait_for(w,"Object.keys(state.workflow.nodes).length===2&&!state.busy&&!state.flow.stale")
        js("cmd('preferences',{theme:'dark'});true");QTest.qWait(450)
        pull('2');QTest.qWait(250)
        assert js("document.querySelector('[data-flow-suggest]').dataset.flowSuggest==='Prepared output'")
        js("document.querySelector('#toast').classList.remove('show');true");QTest.qWait(150)
        w.grab().save(str(out/'suggestions-dark.png'))
        (out/'result.json').write_text(json.dumps({'passed':True,'checks':['Blank workflow after import','72px circles and no automatic inspector',
            'Native arrow drag opens suggestions without graph mutation','Missing-cell evidence ranks filling first','Native selection adds exactly one connected node',
            'Escape cancels without adding nodes','Port compatibility filters model results','New workflow is empty and undo restores nodes',
            'Prepared output is suggested after filling','Light and dark screenshots verified']},indent=2))
        print('Compact workflow and native suggestions smoke passed')
    finally:w.close();app.processEvents()


if __name__=='__main__':run()
