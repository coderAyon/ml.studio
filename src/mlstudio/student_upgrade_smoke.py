"""Native acceptance for the student dashboard, explanations and prediction workspace."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings
from .desktop import ModernWindow
from .ui_smoke import evaluate,wait_for,capture


def run(output=None):
    out=Path(output or 'examples/student-upgrade-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1530,1000);w.show();js=lambda code:evaluate(w,code)
    wait_for(w,"document.body.dataset.ready==='true'")
    original_open,original_save=QFileDialog.getOpenFileName,QFileDialog.getSaveFileName
    checks=[];notices=[];b.notice.connect(lambda data:notices.append(json.loads(data)))
    try:
        capture(w,out,'home');(out/'home.png').replace(out/'home-empty.png')
        rng=np.random.default_rng(23);frame=pd.DataFrame({'signal':rng.normal(size=120),'noise':rng.normal(size=120),'target':rng.integers(0,2,120)})
        source=out/'synthetic-student.csv';frame.to_csv(source,index=False)
        QFileDialog.getOpenFileName=lambda *a:(str(source),'')
        js("studioUI.command('import');true");wait_for(w,'window.appState?.dataset?.rows===120&&!window.appState.busy')
        js("studioUI.navigate('models');document.getElementById('minimal-target').value='target';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
        wait_for(w,"window.appState.target==='target'")
        js("document.querySelector('[data-action=simple_train]').click();true")
        wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
        assert len(c.result.metadata['diagnostics']['records'])==30
        assert js("!!document.querySelector('.upgrade-evidence')")
        js("document.querySelector('.upgrade-evidence').open=true;doAction('prediction_explain');true")
        wait_for(w,'!!window.appState.predictions.explanation&&!window.appState.busy',timeout=60)
        assert len(b.predictions.explanation['rows'])==2
        js("document.querySelector('[data-diagnostic-filter=errors]').click();true")
        assert js("predictionUI.filter==='errors'")
        checks.append('Real CSV import, simple train/test, holdout mistakes, ROC, baseline and held-out input explanation')
        capture(w,out,'home');(out/'home.png').replace(out/'home-loaded.png')
        model=out/'student-model.mlmodel';QFileDialog.getSaveFileName=lambda *a:(str(model),'')
        js("doAction('save_model');true");wait_for(w,"!window.appState.busy")
        assert model.is_file()
        c.result=None;c.dataset=None;c.source=None;b.publish()
        QFileDialog.getOpenFileName=lambda *a:(str(model),'')
        js("doAction('load_model');true");wait_for(w,'!!window.appState.model&&!window.appState.busy')
        assert c.dataset is None
        checks.append('Saved fitted model reopens with no dataset loaded')
        incoming=out/'new-records.csv';new=frame.head(12).rename(columns={'signal':'score'});new.to_csv(incoming,index=False)
        QFileDialog.getOpenFileName=lambda *a:(str(incoming),'')
        js("studioUI.navigate('predictions');document.querySelector('[data-action=student_prediction_import]').click();true")
        wait_for(w,'!!window.appState.predictions.input&&!window.appState.busy')
        js("document.querySelector('[data-prediction-map=signal]').value='score';document.querySelector('[data-prediction-map=signal]').dispatchEvent(new Event('change',{bubbles:true}));document.getElementById('prediction-allow-extra').click();document.querySelector('[data-action=student_prediction_run]').click();true")
        wait_for(w,'!!window.appState.predictions.output&&!window.appState.busy',timeout=60)
        assert b.predictions.summary['rows']==12 and len(b.predictions.summary['probability_columns'])==2
        assert js("!!document.querySelector('.upgrade-prediction-result table')")
        export=out/'predictions.csv';QFileDialog.getSaveFileName=lambda *a:(str(export),'')
        js("doAction('prediction_export');true");wait_for(w,"!window.appState.busy")
        assert len(pd.read_csv(export))==12 and 'prediction' in pd.read_csv(export)
        pd.testing.assert_frame_equal(pd.read_csv(incoming),new.reset_index(drop=True))
        checks.append('Prediction input preview, explicit renamed-column mapping, extra-column opt-in, probabilities and CSV export')
        js("document.querySelector('[data-single-feature=signal]').value='1.2';document.querySelector('[data-single-feature=noise]').value='0.4';doAction('student_prediction_single');true")
        wait_for(w,'!!window.appState.predictions.single')
        assert len(b.predictions.single['table']['data'])==1
        checks.append('One-record prediction without a CSV')
        capture(w,out,'predictions')
        for theme in ['dark','contrast']:
            js("studioUI.command('preferences',{theme:"+json.dumps(theme)+"});true")
            wait_for(w,"document.documentElement.dataset.theme==="+json.dumps(theme))
            capture(w,out,'home');(out/'home.png').replace(out/('home-'+theme+'.png'))
        assert not [n for n in notices if n['kind']=='error'],notices
        import sys
        result={'passed':True,'frozen':bool(getattr(sys,'frozen',False)),'checks':checks+['Light/Dark/High contrast Home and prediction UI','No unexpected engine errors']}
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    finally:
        QFileDialog.getOpenFileName,QFileDialog.getSaveFileName=original_open,original_save
        if not c.busy:w.close()
        app.processEvents()


if __name__=='__main__':run()
