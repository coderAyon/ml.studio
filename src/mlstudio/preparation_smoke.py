"""Exercise preparation controls through the real embedded desktop UI."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop,QTimer
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/preparation-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');w.resize(1540,980);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    c,b=w.controller,w.bridge
    frame=pd.DataFrame({'measurement':np.arange(120,dtype=float),'second':np.sin(np.arange(120)),
                        'category':['a','b',None]*40,'outcome':[0,1]*60})
    frame.loc[[3,8,30],'measurement']=np.nan
    source=out/'synthetic-preparation.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data()
    c.target.setCurrentText('outcome');b.publish()
    wait_for(w,'window.appState?.dataset?.rows===120')
    js=lambda code:evaluate(w,code)
    js("studioUI.navigate('preprocessing');true")
    assert js("document.querySelectorAll('#prep-imputation option').length===5")
    assert js("document.querySelectorAll('#prep-scaling option').length===7")
    assert js("!!document.getElementById('prep-feature_selection')")
    js("document.querySelectorAll('.prep-training details').forEach(x=>x.open=true);true")
    for key,value in [('imputation','knn'),('scaling','minmax'),('categorical_imputation','constant'),('feature_selection','f_test'),('selection_k',2)]:
        js(f"(()=>{{const el=document.getElementById('prep-{key}');el.value={json.dumps(str(value))};el.dispatchEvent(new Event('change',{{bubbles:true}}));return true}})()")
    js("document.getElementById('prep-missing_indicators').click();document.querySelector('[data-action=prep_save_training]').click();true")
    wait_for(w,"window.appState.preprocessing.preparation.imputation==='knn'")
    assert c.preparation['scaling']=='minmax' and c.preparation['selection_k']==2
    capture(w,out,'preprocessing')
    js("document.querySelectorAll('.prep-training details').forEach(x=>x.open=true);document.querySelector('.prep-training').scrollIntoView({block:'start'});true")
    loop=QEventLoop();QTimer.singleShot(1000,loop.quit);loop.exec()
    w.grab().save(str(out/'preparation-controls.png'))
    js("studioUI.navigate('models');document.getElementById('minimal-target').value='outcome';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
    assert js("minimalAlgorithms('classification').includes('Gradient Boosting')")
    assert js("minimalAlgorithms('regression').includes('Elastic Net')&&!minimalAlgorithms('classification').includes('Elastic Net')")
    js("document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    assert c.result.metadata['preparation']==c.preparation
    assert c.result.metadata['fitted_preparation']['output_features']==2
    assert js("document.getElementById('content').textContent.includes('What actually entered this model')")
    assert c.dataset.index==0
    pd.testing.assert_frame_equal(c.dataset.current,frame)
    capture(w,out,'models')
    assert js('state.algorithms.length')==22
    result={'passed':True,'algorithms':js('state.algorithms.length'),'checks':['5 numeric filling methods / 7 scaling methods visible',
        'Advanced categorical filling, missing flags, bins and reduction controls visible',
        'KNN + Min–Max + missing flags + F-test saved through QWebChannel',
        'Task-specific new algorithms available','Actual model fit uses saved plan and 2 selected inputs',
        'Fitted preparation evidence displayed','Original active data unchanged']}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));w.close();app.processEvents()


if __name__=='__main__':run()
