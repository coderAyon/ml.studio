"""Native checks for wide-column preview, guided train/test and round workflow interactions."""
import json
from pathlib import Path
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings,QEventLoop,QTimer,Qt,QPoint
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/easy-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel');w.resize(1540,980);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    def js(code):return evaluate(w,code)
    def pause(ms=200):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    checks=[]
    frame=pd.DataFrame({**{'feature_'+str(i):[(j+i)%31 for j in range(120)] for i in range(12)},
                        'score':[j*.27+.3 for j in range(120)],'outcome':[0,1]*60})
    source=out/'wide-qa.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    wait_for(w,'window.appState?.dataset?.rows===120')
    navigation=js("(()=>{try{studioUI.navigate('data');return view.page}catch(e){return String(e.stack)}})()")
    assert navigation=='data',navigation
    assert js("document.querySelectorAll('.easy-dataset-table thead th').length===16"),js("JSON.stringify({columns:state.dataset.columns,headers:Array.from(document.querySelectorAll('.easy-dataset-table th')).map(n=>n.textContent),text:content.textContent.slice(0,250)})")
    assert js("document.querySelector('.easy-dataset-table .table-scroll').scrollWidth>document.querySelector('.easy-dataset-table .table-scroll').clientWidth")
    js("document.querySelector('[data-action=easy_columns_first]').click();true")
    assert js("document.querySelectorAll('.easy-dataset-table thead th').length===7")
    js("document.getElementById('easy-column-search').value='feature_1';document.getElementById('easy-column-search').dispatchEvent(new Event('input',{bubbles:true}));true")
    assert js("document.querySelectorAll('[data-column-choice]:not([hidden])').length===3")
    js("document.querySelector('[data-action=easy_columns_matching]').click();true")
    assert js("document.querySelectorAll('.easy-dataset-table thead th').length===9")
    js("document.querySelector('[data-action=easy_columns_all]').click();true")
    assert js("document.querySelectorAll('.easy-dataset-table thead th').length===16")
    js("document.querySelector('[data-action=easy_expand]').click();true")
    assert js("document.body.classList.contains('easy-table-expanded')")
    js("document.querySelector('.easy-dataset-table .table-scroll').scrollLeft=3000;true")
    assert js("document.querySelector('.easy-dataset-table .table-scroll').scrollLeft>0")
    w.raise_();w.activateWindow();pause(1000);w.grab().save(str(out/'full-preview.png'))
    js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));true")
    assert js("!document.body.classList.contains('easy-table-expanded')")
    checks.append('All 14 columns, searchable column selection, Show all, sideways scroll and expanded preview')
    js("studioUI.navigate('models');document.getElementById('minimal-target').value='score';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
    wait_for(w,"window.appState.target==='score'")
    assert js("minimalModel.task==='regression'")
    js("document.getElementById('minimal-target').value='outcome';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
    wait_for(w,"window.appState.target==='outcome'")
    assert js("minimalModel.task==='classification'")
    js("document.getElementById('easy-test-size').value='0.2';document.getElementById('easy-test-size').dispatchEvent(new Event('change',{bubbles:true}));true")
    assert js("document.querySelector('.easy-training-plan').textContent.includes('96')")
    js("document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    assert c.result.metadata['training_rows']==96 and c.result.metadata['test_rows']==24
    capture(w,out,'models')
    checks.append('Reviewable goal suggestion and one-click real train/test with chosen 80/20 share')
    js("studioUI.navigate('workflow');document.querySelector('.library-node[data-kind=Dataset]').click();true")
    wait_for(w,'Object.keys(window.appState.workflow.nodes).length===1')
    js("document.querySelector('.library-node[data-kind=\"Remove duplicates\"]').click();true")
    wait_for(w,'Object.keys(window.appState.workflow.nodes).length===2')
    js("document.querySelector('.library-node[data-kind=\"Train model\"]').click();true")
    wait_for(w,'Object.keys(window.appState.workflow.nodes).length===3')
    assert len(c.canvas.graph.edges)==2
    assert js("getComputedStyle(document.querySelector('.workflow-node')).borderRadius==='50%'")
    assert js("Math.abs(document.querySelector('.workflow-node').getBoundingClientRect().width-document.querySelector('.workflow-node').getBoundingClientRect().height)<1")
    # Disconnect and reconnect with whole-node clicks rather than tiny ports.
    js("doAction('disconnect');true")
    wait_for(w,'window.appState.workflow.edges.length===1')
    js("doAction('easy_flow_connect');document.querySelector('[data-node=\"2\"]').click();document.querySelector('[data-node=\"3\"]').click();true")
    wait_for(w,'window.appState.workflow.edges.length===2')
    assert c.canvas.graph.order()==['1','2','3']
    js("studioUI.command('run_workflow');true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    capture(w,out,'workflow')
    checks.append('Click-to-add round nodes, automatic edges, whole-node reconnect and actual workflow execution')
    js("document.querySelector('[data-node=\"1\"]').scrollIntoView({block:'center',inline:'center'});true")
    pause(300)
    position=json.loads(js("JSON.stringify((()=>{const r=document.querySelector('[data-node=\"1\"]').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})())"))
    before=list(c.canvas.graph.nodes['1']['position'])
    surface=w.view.focusProxy() or w.view
    start=QPoint(round(position['x']),round(position['y']))
    QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(surface,start+QPoint(35,24),100)
    QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start+QPoint(35,24))
    wait_for(w,"window.appState.workflow.nodes['1'].position[0]>"+str(before[0]+10))
    checks.append('Native mouse drag moves round nodes and saves their new graph positions')
    # Existing native drag/drop handlers continue adding nodes with correct coordinates.
    js("(()=>{const host=document.querySelector('.canvas-host'),rect=host.getBoundingClientRect(),transfer=new DataTransfer();transfer.setData('text/plain','Dataset');host.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer,clientX:rect.left+100,clientY:rect.top+200}));return true;})()")
    wait_for(w,'Object.keys(window.appState.workflow.nodes).length===4')
    checks.append('Existing drag/drop node creation remains available')
    js("studioUI.command('preferences',{theme:'dark'});true")
    wait_for(w,"document.documentElement.dataset.theme==='dark'")
    capture(w,out,'workflow');(out/'workflow.png').replace(out/'workflow-dark.png')
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
    w.close();app.processEvents()


if __name__=='__main__':run()
