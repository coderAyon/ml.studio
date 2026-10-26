"""Visual and navigation acceptance for the quiet Home page."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .training import train
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/home-quiet-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel');w.resize(1530,1000);w.show()
    js=lambda code:evaluate(w,code)
    wait_for(w,"document.body.dataset.ready==='true'")
    assert js("!!document.querySelector('.home-quiet')")
    assert js("document.querySelectorAll('#content .panel,#content .stat-card,#content .minimal-metric').length===0")
    assert js("document.querySelector('.hq-dataset').textContent.includes('Bring your data')")
    capture(w,out,'home');(out/'home.png').replace(out/'light-empty.png')
    f=pd.DataFrame({'length':np.arange(80,dtype=float),'width':np.sin(np.arange(80)), 'outcome':[0,1]*40})
    f.loc[2,'length']=np.nan
    source=out/'synthetic-home.csv';f.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('outcome')
    c.training_done(train(f,'outcome','Decision Tree'));b.publish()
    wait_for(w,'window.appState?.dataset?.rows===80')
    assert js("document.querySelector('.hq-metrics').textContent.includes('80')")
    assert js("document.querySelector('.hq-metrics').textContent.includes('Missing cells1')")
    assert js("document.querySelector('.hq-run-list').textContent.includes('Decision Tree')")
    assert js("document.querySelector('.hq-model-line').textContent.includes('outcome')")
    capture(w,out,'home');(out/'home.png').replace(out/'light-loaded.png')
    js("document.querySelector('.hq-value-link').click();true")
    wait_for(w,"!!document.querySelector('#modal-root [role=dialog]')")
    assert js("document.querySelector('#modal-root').textContent.includes('Missing')")
    js("beginner.issuePopup=false;render();document.querySelector('.hq-run-list button').click();true")
    wait_for(w,"!!document.querySelector('#modal-root [role=dialog]')")
    assert js("document.querySelector('#modal-root').textContent.includes('Decision Tree')")
    js("view.experiment=null;render();true")
    for page in ['data','preprocessing','dataset_visuals','models']:
        js("document.querySelector('.hq-step[data-page="+page+"]').click();true")
        assert js('view.page')==page
        js("studioUI.navigate('home');true")
    for theme in ['dark','contrast']:
        js("studioUI.command('preferences',{theme:"+json.dumps(theme)+"});true")
        wait_for(w,"document.documentElement.dataset.theme==="+json.dumps(theme))
        capture(w,out,'home');(out/'home.png').replace(out/(theme+'-loaded.png'))
    w.resize(1050,850);capture(w,out,'home');(out/'home.png').replace(out/'compact.png')
    assert js("document.documentElement.scrollWidth<=innerWidth")
    assert js("document.querySelector('.home-quiet').scrollWidth<=document.querySelector('.home-quiet').clientWidth")
    pd.testing.assert_frame_equal(f,c.dataset.current)
    result={'passed':True,'checks':['Empty and loaded Home','No repeated panel/stat cards','Actual dataset counts and model status',
        'Quality popup and experiment inspection','All four workspace links','Light/Dark/High contrast','Compact layout without horizontal overflow','Original data unchanged']}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    w.close();app.processEvents()


if __name__=='__main__':run()
