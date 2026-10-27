"""Cross-dataset desktop acceptance for identity, selection, cleaning and preparation."""
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
    out=Path('examples/workspace-matrix-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    w.resize(1700,1050);w.show();js=lambda s:evaluate(w,s)
    source=out/'same-path-different-datasets.csv'
    def load(f):
        f.to_csv(source,index=False);c.source=str(source);c.dataset=DatasetVersions(f);c.result=None;c.preparation=None;c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('outcome');b.publish();QTest.qWait(300)
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        f=pd.DataFrame({'value':np.arange(40,dtype=float),'category':['Dhaka','ঢাকা']*20,'outcome':['yes','no']*20});f.index=['repeat']*40
        load(f);js("studioUI.navigate('data');true")
        js("document.querySelector('[data-select-row=\"1\"]').click();true")
        assert js("document.querySelectorAll('[data-select-row]:checked').length===1&&explorer.selected.size===1")
        js("document.querySelector('[data-select-row=\"4\"]').click();document.querySelector('[data-integrity-selection]').click();true")
        wait_for(w,'state.preprocessing.preview?.summary.after_rows===2')
        assert b.preprocess_preview['summary']['removed_rows']==38
        js("document.querySelector('[data-action=preprocess_apply]').click();true")
        wait_for(w,'state.dataset.rows===2')
        assert c.dataset.current.value.tolist()==[0.,3.]
        js("cmd('undo');true");wait_for(w,'state.dataset.rows===40')
        js("studioUI.navigate('data');explorer.hidden.add('category');explorer.sort='value';explorer.page=3;view.search='old search';explorer.selected.add(1);render();true")
        first=b.dataset_state()['instance_key']
        g=pd.DataFrame({'new_category':['A',None,'B','C']*10,'outcome':['yes','no']*20});load(g)
        assert first!=b.dataset_state()['instance_key']
        assert js("explorer.selected.size===0&&explorer.hidden.size===0&&view.search===''&&explorer.page===0&&explorer.sort===''")
        js("studioUI.navigate('models');minimalModel.target='outcome';minimalModel.algorithm='Decision Tree';cmd('simple_train',{target:'outcome',task:'classification',algorithm:'Decision Tree'});true")
        wait_for(w,'!!state.model&&!state.busy',timeout=60)
        assert c.result.metadata['features']==['new_category']
        # A third dataset has a nullable integer whose median is fractional.
        h=pd.DataFrame({'nullable':pd.Series([1,2,None,None]*10,dtype='Int64'),'outcome':['yes','no']*20});load(h)
        js("studioUI.navigate('preprocessing');cmd('preprocess_preview',{operation:'Fill missing',column:'nullable',value:''});true")
        wait_for(w,"state.preprocessing.preview?.summary.after_missing===0")
        assert b.preprocess_preview['summary']['changed_cells']==20
        assert b.preprocess_preview['after']['data'][0][0]==1.5
        js("document.querySelector('.prep-review').scrollIntoView({block:'center'});document.querySelector('#toast').classList.remove('show');true");QTest.qWait(500)
        w.grab().save(str(out/'nullable-cleaning-preview.png'))
        js("studioUI.navigate('settings');true")
        assert js("document.querySelector('#content').textContent.includes('200 retained inputs')")
        assert js("!document.querySelector('#content').textContent.includes('persistent model bundles, general-purpose')")
        (out/'result.json').write_text(json.dumps({'passed':True,'datasets':['Mixed numeric/Bengali category with duplicate index labels',
            'Different categorical-only CSV reusing the same source path','Nullable integer with missing values'],
            'checks':['Row selection uses active positions','Keep-selected preview matches apply','Undo restores full dataset',
                'Same-path reimport resets old visibility/search/sort/selection','Category-only dataset trains successfully',
                'Nullable integer cleaning preview fills exact gaps','Settings disclose actual tools and limits']},indent=2))
        print('Cross-dataset workspace desktop smoke passed')
    finally:w.close();app.processEvents()


if __name__=='__main__':run()
