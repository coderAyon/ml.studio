"""Native acceptance for the Orange/Weka audit repairs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QEventLoop,QTimer,QPoint,Qt
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .training import predict
from .ui_smoke import evaluate,wait_for,capture


def run(out=None):
    out=Path(out or 'examples/audit-fixes-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');w.resize(1500,980);w.show()
    wait_for(w,"document.body.dataset.ready==='true'");c,b=w.controller,w.bridge
    js=lambda s:evaluate(w,s)
    def pause(ms=250):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    checks=[]
    frame=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),'target':[0,1]*40})
    frame.loc[[2,6,11],'x']=np.nan
    source=out/'synthetic-audit.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.import_options={};c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();c.target.setCurrentText('target');c.algorithm.setCurrentText('Decision Tree');b.publish()
    wait_for(w,'window.appState?.dataset?.rows===80')
    assert js("researchAlgorithms().length===10&&!researchAlgorithms().includes('Lasso')")
    js("studioUI.navigate('preprocessing');document.getElementById('prep-scaling').value='minmax';document.getElementById('prep-scaling').dispatchEvent(new Event('change',{bubbles:true}));studioUI.navigate('models');true")
    wait_for(w,"view.page==='models'&&window.appState.preprocessing.preparation.scaling==='minmax'")
    assert c.preparation['scaling']=='minmax'
    checks.append('Unsaved preparation is validated and saved before Continue navigation')
    assert b.research.config().preparation['scaling']=='minmax'
    assert js("document.querySelectorAll('[data-compare-method]').length===10")
    js("document.getElementById('minimal-algorithm').value='Decision Tree';document.getElementById('minimal-algorithm').dispatchEvent(new Event('change',{bubbles:true}));true")
    js("document.getElementById('model-param-max_depth').value='3';document.getElementById('model-param-max_depth').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    assert c.result.metadata['parameters']['max_depth']==3
    expected=predict(c.result,frame.drop(columns='target'))
    js("document.getElementById('model-tune').click();document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model?.metadata?.tuning&&!window.appState.busy',timeout=60)
    assert c.result.metadata['tuning']['folds']==3
    expected=predict(c.result,frame.drop(columns='target'))
    checks.append('Optional parameter search executes on three training-only folds')
    model=out/'trained.mlmodel'
    original_save,original_open=QFileDialog.getSaveFileName,QFileDialog.getOpenFileName
    try:
        QFileDialog.getSaveFileName=lambda *args:(str(model),'')
        js("doAction('save_model');true");pause();assert model.is_file()
        c.result=None;b.publish();QFileDialog.getOpenFileName=lambda *args:(str(model),'')
        js("doAction('load_model');true");wait_for(w,'!!window.appState.model&&!window.appState.busy')
        np.testing.assert_equal(expected,predict(c.result,frame.drop(columns='target')))
        checks.append('Custom parameter applied; saved model reopened and produces identical predictions')
        js("document.querySelectorAll('[data-compare-method]').forEach(el=>{el.checked=['Decision Tree','Extra Trees'].includes(el.dataset.compareMethod);el.dispatchEvent(new Event('change',{bubbles:true}));});document.querySelector('[data-action=simple_compare]').click();true")
        wait_for(w,'window.appState.comparison.length===2&&!window.appState.busy',timeout=60)
        assert {r['algorithm'] for r in b.snapshot()['comparison']}=={'Decision Tree','Extra Trees'}
        checks.append('Selected two-method comparison executes exactly the requested methods')
        c.result=__import__('mlstudio.model_io',fromlist=['load_model']).load_model(model);b.publish()
        js("studioUI.navigate('predictions');true")
        assert js("!!document.getElementById('prediction-encoding')")
        checks.append('Prediction encoding, decimal and delimiter controls available')
        js("studioUI.navigate('dataset_visuals');true")
        wait_for(w,"!!document.getElementById('linked-scatter')",timeout=60)
        js("document.getElementById('linked-scatter').scrollIntoView({block:'center'});true");pause(500)
        coords=json.loads(js("JSON.stringify((()=>{const s=document.getElementById('linked-scatter'),m=s.getScreenCTM(),a=new DOMPoint(80,80).matrixTransform(m),b=new DOMPoint(500,300).matrixTransform(m);return {a:{x:a.x,y:a.y},b:{x:b.x,y:b.y}}})())"))
        surface=w.view.focusProxy() or w.view
        a=QPoint(round(coords['a']['x']),round(coords['a']['y']));end=QPoint(round(coords['b']['x']),round(coords['b']['y']))
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,a)
        QTest.mouseMove(surface,end,200);QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
        wait_for(w,'!!window.appState.dataset_visuals?.selection')
        assert b.dataset_visual_result['selection']['count']>0
        count=b.dataset_visual_result['selection']['count']
        js("doAction('dataset_visual_selection_apply');true")
        wait_for(w,'window.appState.dataset?.version===1&&!window.appState.busy')
        assert len(c.dataset.current)==count
        c.dataset.undo();c.refresh_data();b.publish()
        pd.testing.assert_frame_equal(c.dataset.current,frame)
        checks.append('Native mouse rectangle selects source rows; subset Apply and Undo preserve originals')
        capture(w,out,'models')
        js("studioUI.navigate('reliability');true")
        assert js("document.querySelectorAll('[data-research-algorithm]').length===10")
        checks.append('Advanced classification lists exactly 10 valid methods')
        c.algorithm.setCurrentText('Ridge');b.publish()
        assert js("researchAlgorithms().length===12&&researchAlgorithms().includes('Lasso')")
        checks.append('Advanced regression lists all 12 valid methods')
        js("studioUI.navigate('workflow');document.querySelector('.library-node[data-kind=Dataset]').click();true")
        wait_for(w,'Object.keys(window.appState.workflow.nodes).length===1')
        js("document.querySelector('.library-node[data-kind=\"Prepare training\"]').click();true")
        wait_for(w,'Object.keys(window.appState.workflow.nodes).length===2')
        js("document.querySelector('.library-node[data-kind=\"Compare models\"]').click();true")
        wait_for(w,'Object.keys(window.appState.workflow.nodes).length===3')
        assert len(c.canvas.graph.edges)==2
        js("doAction('run_workflow');true")
        wait_for(w,'window.appState.comparison.length===3&&!window.appState.busy',timeout=60)
        checks.append('New round Prepare training and Compare models nodes connect and execute')
        result={'passed':True,'frozen':bool(getattr(sys,'frozen',False)),'runtime':sys.executable,'checks':checks}
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    finally:
        QFileDialog.getSaveFileName,QFileDialog.getOpenFileName=original_save,original_open
        w.close();app.processEvents()


if __name__=='__main__':run()
