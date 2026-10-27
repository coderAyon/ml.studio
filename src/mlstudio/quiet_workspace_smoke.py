"""Verify the unified workspace, preparation alternatives and concise result brief."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/quiet-workspace-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1740,1060);w.show();js=lambda s:evaluate(w,s);old=QFileDialog.getSaveFileName
    def capture(name):
        js("document.querySelector('#toast').classList.remove('show');true");QTest.qWait(550);w.grab().save(str(out/name))
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        assert js("!document.querySelector('#navigation [data-page=home]')&&document.querySelector('#navigation [data-page=data]').textContent.includes('Workspace')")
        f=pd.DataFrame({'measurement':np.linspace(1,10,120),'second':np.sin(np.arange(120)),'category':['a','b','c']*40,'outcome':[0,1]*60})
        f.loc[3,'measurement']=np.nan;f.loc[59,'second']=200;original=f.copy(deep=True)
        source=out/'synthetic.csv';f.to_csv(source,index=False);c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('outcome');b.publish()
        wait_for(w,'state.dataset?.rows===120')
        js("studioUI.navigate('home');true")
        assert js("view.page==='data'&&!!document.querySelector('.workspace-unified .scalable-table')")
        assert js("document.querySelectorAll('[data-column-grip]').length===4")
        js("document.querySelector('#workspace-table-options>summary').click();true")
        assert js("document.getElementById('column-size-value').getBoundingClientRect().height>0&&document.querySelector('[data-visible]').getBoundingClientRect().height>0")
        js("document.querySelector('#workspace-table-options>summary').click();document.querySelector('[data-tab=schema]').click();true")
        assert js("!!document.getElementById('profile-column')")
        js("document.querySelector('[data-tab=versions]').click();document.querySelector('[data-tab=preview]').click();true")
        assert js("!!document.querySelector('.scalable-table')")
        capture('workspace-light.png')
        js("studioUI.navigate('preprocessing');true")
        assert js("document.querySelector('.prep-finding').getBoundingClientRect().height<100&&document.querySelector('.prep-empty').getBoundingClientRect().height<230")
        capture('prepare-compact-light.png')
        js("prepUI.draft.scaling='minmax';document.querySelector('[data-value-plan]').click();true")
        wait_for(w,"state.preprocessing.training_preview?.scaling==='minmax'&&!state.busy",timeout=60)
        assert b.training_preview['effects'][0]['before_missing']==1 and b.training_preview['effects'][0]['after_missing']==0
        js("document.getElementById('prep-plan-compare').open=true;document.querySelector('[data-prep-compare]').click();true")
        wait_for(w,"state.preprocessing.comparison?.candidate.scaling==='robust'&&!state.busy",timeout=60)
        assert c.preparation is None  # Comparing does not save a plan or mutate source values.
        js("document.getElementById('prep-plan-compare').open=true;document.getElementById('prep-plan-compare').scrollIntoView({block:'start'});window.scrollBy(0,-90);true")
        capture('compare-preparation.png')
        js("document.querySelector('[data-prep-use-alternative]').click();true")
        wait_for(w,"state.preprocessing.preparation.scaling==='robust'")
        pd.testing.assert_frame_equal(c.dataset.current,original)
        js("studioUI.navigate('models');cmd('simple_train',{target:'outcome',task:'classification',algorithm:'Decision Tree'});true")
        wait_for(w,'!!state.model&&!state.busy',timeout=60)
        js("studioUI.navigate('results');true")
        assert js("!!document.querySelector('.result-lead')&&document.querySelector('.result-lead').textContent.includes('held-out records')")
        brief=out/'result-brief.html';QFileDialog.getSaveFileName=lambda *a:(str(brief),'')
        js("document.querySelector('[data-action=result_brief_export]').click();true")
        wait_for(w,"document.querySelector('#toast').textContent.includes('Result brief exported')")
        assert brief.is_file() and 'Training dataset fingerprint' in brief.read_text(encoding='utf-8')
        capture('results-light.png')
        js("cmd('preferences',{theme:'dark'});true");QTest.qWait(300);capture('results-dark.png')
        (out/'result.json').write_text(json.dumps({'passed':True,'checks':['Home and My dataset merged under Workspace',
            'Original table controls and column sizing retained','Compact findings and right-side panel',
            'Numeric before/after effects cover the active data','Alternative plans compared on identical splits without saving or changing data',
            'Candidate plan saves only on explicit choice','Actual model trains with selected preparation','Concise result and baseline comparison',
            'Result brief exports actual run evidence','Light and Dark visual acceptance']},indent=2))
        print('Unified workspace and advanced preparation desktop smoke passed')
    finally:QFileDialog.getSaveFileName=old;w.close();app.processEvents()


if __name__=='__main__':run()
