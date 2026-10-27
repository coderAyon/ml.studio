"""Exercise before/after preview through the real embedded desktop bridge."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/value-preview-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');w.resize(1600,1000);w.show()
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        c,b=w.controller,w.bridge
        f=pd.DataFrame({'measurement':np.arange(80,dtype=float)%10,'second':np.arange(80,dtype=float),'outcome':[0,1]*40})
        f.loc[2,'measurement']=100
        source=out/'synthetic.csv';f.to_csv(source,index=False)
        c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('outcome');b.publish()
        js=lambda s:evaluate(w,s)
        wait_for(w,'state.dataset?.rows===80')
        js("studioUI.navigate('preprocessing');prepUI.draft.scaling='minmax';document.querySelector('[data-value-plan]').click();true")
        wait_for(w,"!!document.querySelector('.prep-review .value-comparison')")
        js("document.querySelector('.value-row').value='3';document.querySelector('.value-column').value='measurement';document.querySelector('[data-value-refresh]').click();true")
        wait_for(w,"state.preprocessing.training_preview?.selected_row===3")
        cells=b.training_preview['cells'];assert len(cells)==1 and cells[0]['before']==100 and cells[0]['after']!=100
        assert js("document.querySelector('.prep-review').textContent.includes('IQR outlier')")
        js("document.querySelector('.prep-review').scrollIntoView({block:'start'});true")
        QTest.qWait(800)
        w.grab().save(str(out/'preparation-before-after.png'))
        graph={'nodes':{'1':{'type':'Dataset','position':[50,100],'params':{'source':'workspace'}},'2':{'type':'Normalize','position':[300,100],
            'params':{'scaling':'minmax','preview_row':3,'preview_column':'measurement'}}},'edges':[['1','2']]}
        js("studioUI.navigate('workflow');cmd('flow_commit',{graph:"+json.dumps(graph)+",auto:true});true")
        wait_for(w,"state.flow.outputs['2']?.comparison?.selected_row===3&&!state.busy",timeout=60)
        js("flowSelect('2');true")
        assert js("document.querySelector('.wf-output .value-comparison').textContent.includes('100')")
        QTest.qWait(800)
        js("document.querySelector('#toast').classList.remove('show');true")
        QTest.qWait(300)
        w.grab().save(str(out/'workflow-before-after.png'))
        assert c.dataset.current.iloc[2,0]==100
        (out/'result.json').write_text(json.dumps({'passed':True,'checks':['Right-side preparation preview','Exact row 3: original 100 and transformed value',
            'Original outlier remains marked','Workflow Normalize shows before/after immediately','CSV remains unchanged'],'cell':cells[0]},indent=2))
        print('Value preview desktop smoke passed')
    finally:w.close();app.processEvents()


if __name__=='__main__':run()
