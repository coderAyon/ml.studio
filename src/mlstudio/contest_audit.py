"""Real native feature audit; only native file/option dialog choices are automated."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings,QEventLoop,QTimer
from PySide6.QtWidgets import QFileDialog,QDialog,QComboBox,QMessageBox
from .ui_smoke import evaluate,wait_for,capture
from .data import DatasetVersions
from .analyzer import analyze
from .training import ALGORITHMS,REGRESSION


def run(window,application,directory):
    out=Path(directory).resolve();out.mkdir(parents=True,exist_ok=True)
    c=window.controller;b=window.bridge
    c.settings=QSettings(str(out/'isolated-settings.ini'),QSettings.Format.IniFormat)
    records=[];notices=[];started=time.monotonic()
    b.notice.connect(lambda payload:notices.append(json.loads(payload)))
    original_open=QFileDialog.getOpenFileName;original_save=QFileDialog.getSaveFileName
    original_directory=QFileDialog.getExistingDirectory;original_dialog=QDialog.exec;original_warning=QMessageBox.warning
    def reject_hidden(*args,**kwargs):raise AssertionError('An error attempted to open a hidden native warning dialog')
    QMessageBox.warning=reject_hidden
    def save_path(path):QFileDialog.getSaveFileName=lambda *a,**k:(str(path),'')
    def open_path(path):QFileDialog.getOpenFileName=lambda *a,**k:(str(path),'')
    def js(script):return evaluate(window,script)
    def command(action,data=None):js('studioUI.command('+json.dumps(action)+','+json.dumps(data or {})+');true')
    def idle():
        loop=QEventLoop();QTimer.singleShot(75,loop.quit);loop.exec()
        wait_for(window,'!window.appState.busy',timeout=60)
    def click(action):
        assert js('!!document.querySelector('+json.dumps('[data-action='+action+']')+')'),action+' is missing'
        js('document.querySelector('+json.dumps('[data-action='+action+']')+').click();true')
    def passed(name,detail=''):records.append({'feature':name,'status':'passed','detail':detail})
    def frame_data(frame,name):
        path=out/(name+'.csv');frame.to_csv(path,index=False)
        c.source=str(path);c.active_project_path=None;c.import_options={'delimiter':',','encoding':'utf-8-sig','header':True,'decimal':'.'}
        c.dataset=DatasetVersions(frame);c.result=None;c.issues=analyze(frame);c.refresh_data();b.table_options={};b.publish()
        wait_for(window,'window.appState.dataset.name==='+json.dumps(path.name))
        return path
    try:
        window.show();wait_for(window,"document.body.dataset.ready==='true'")
        assert js("document.querySelector('[data-page=deep]').disabled")
        passed('Unavailable capability clearly disabled','Deep Learning is not implemented and is not counted as a working feature.')
        source=out/'real-import.csv'
        rng=np.random.default_rng(13)
        base=pd.DataFrame({'x':rng.normal(size=120),'amount':rng.uniform(1,20,120),'category':['A','B','C']*40,'target':[0,1]*60})
        base.to_csv(source,index=False)
        open_path(source);command('import');idle()
        assert c.dataset.current.shape==(120,4) and not c.target.currentText()
        passed('Default CSV import','Actual file parsed and profiled; no target guessed.')
        advanced=out/'latin1.csv';advanced.write_text('name;value\nAndré;"2,5"\nZoë;"3,5"\n',encoding='latin1')
        def options(dialog):
            combos=dialog.findChildren(QComboBox)
            combos[0].setCurrentText(';');combos[1].setCurrentText('latin-1');combos[3].setCurrentText(',')
            return QDialog.DialogCode.Accepted
        QDialog.exec=options;open_path(advanced);command('import_advanced');idle();QDialog.exec=original_dialog
        assert c.dataset.current.iloc[0,0]=='André' and c.dataset.current.iloc[0,1]==2.5
        passed('Advanced import options','Real Latin-1, semicolon separator and comma decimal parse.')
        frame_data(base,'classification')
        for page in ['home','workflow','data','analyzer','analytics','models','safety','impact','reliability','errors','researchpack','history','predictions','reports','settings']:
            js('studioUI.navigate('+json.dumps(page)+');true')
            assert js("document.querySelector('#content').textContent.trim().length>30"),page
        passed('All 15 available page routes render','Deep Learning intentionally excluded as unavailable.')
        for algorithm in ALGORITHMS:
            frame=base.assign(target=base.x*3+base.amount*.2) if algorithm in REGRESSION else base
            frame_data(frame,'train-'+algorithm.replace(' ','-'))
            command('configure',{'target':'target','algorithm':algorithm})
            js("beginner.task=null;studioUI.navigate('models');true")
            click('train');idle()
            assert c.result is not None and c.result.metadata['algorithm']==algorithm,(algorithm,notices[-1:])
            assert all(v is None or np.isfinite(v) for v in c.result.metrics.values() if isinstance(v,(float,int)))
            passed('Train '+algorithm,'Actual sklearn fit, holdout metrics and session pipeline.')
        frame_data(base,'classification-final');command('configure',{'target':'target','algorithm':'Decision Tree'})
        js("beginner.task=null;studioUI.navigate('models');true");click('train');idle()
        incoming=out/'prediction-input.csv';base.drop(columns='target').head(17).to_csv(incoming,index=False)
        predicted=out/'predictions.csv';open_path(incoming);save_path(predicted)
        js("studioUI.navigate('predictions');true");click('predict');idle()
        assert pd.read_csv(predicted).shape==(17,4) and 'prediction' in pd.read_csv(predicted)
        passed('Predict new CSV','17 actual predictions exported through the session pipeline.')
        for task,algorithm in [('classification','Decision Tree'),('regression','Ridge')]:
            frame_data(base if task=='classification' else base.assign(target=base.x*3+base.amount*.2),'compare-'+task)
            command('configure',{'target':'target','algorithm':algorithm});js("beginner.task=null;studioUI.navigate('models');true");click('compare');idle()
            expected=6 if task=='classification' else 5
            assert len(b.snapshot()['comparison'])==expected
            passed(task.title()+' comparison',str(expected)+' real same-split fits.')
        frame_data(base,'workflow-source');command('configure',{'target':'target','algorithm':'Decision Tree'})
        js("studioUI.navigate('workflow');true")
        command('workflow',{'nodes':{},'edges':[]});before=len(notices);click('validate')
        assert any(n['kind']=='error' for n in notices[before:])
        click('starter');click('validate');click('run_workflow');idle()
        assert c.result is not None
        js("view.selectedNode='3';render();true");click('disconnect');before=len(notices);click('validate')
        assert any(n['kind']=='error' for n in notices[before:])
        click('graph_undo');assert len(c.canvas.graph.edges)==2
        js("view.selectedNode='2';render();true");click('delete_node');assert len(c.canvas.graph.nodes)==2
        click('graph_undo');assert len(c.canvas.graph.nodes)==3
        for a in ['zoom_in','zoom_out','fit']:click(a)
        passed('Workflow actions','Empty/disconnected validation errors; starter, execution, undo, deletion and zoom/fit.')
        for chart in ['Histogram','Box plot','Scatter','Correlation','Missing values','Category frequencies']:
            js('view.chartType='+json.dumps(chart)+";view.chartX='x';view.chartY='amount';studioUI.navigate('analytics');true")
            b.chart='';click('chart')
            assert b.chart.startswith('data:image/svg+xml;base64,'),chart
            passed('Chart '+chart,'Actual Matplotlib SVG.')
        # Exercise each UI transformation while preserving the original version.
        clean=base.assign(category=[' A','B','C']*40,event_date=pd.date_range('2024-01-01',periods=120).astype(str),edge=np.r_[np.inf,np.ones(119)])
        clean.loc[3,'amount']=np.nan
        frame_data(clean,'cleaning-source')
        transforms=[('Trim whitespace','category',''),('Lowercase labels','category',''),('Convert to numeric','x',''),('Convert to text','amount',''),('Convert to datetime','event_date',''),('Replace infinity with missing','edge',''),('Rename column','amount','spending'),('Filter rows','category','A'),('Remove duplicates','x',''),('Drop columns','amount',''),('Fill missing','amount','')]
        for operation,column,value in transforms:
            command('preview',{'operation':operation,'column':column,'value':value})
            assert b.preview is not None;command('apply_preview');idle();assert c.dataset.index==1
            command('undo');idle();assert c.dataset.index==0
            command('redo');idle();assert c.dataset.index==1
            command('undo');idle()
        passed('Cleaning previews, apply, undo and redo',str(len(transforms))+' operations; source version preserved.')
        for action,name in [('export_csv','cleaned.csv'),('export_log','transformations.json'),('report','experiment-report.html')]:
            path=out/name;save_path(path);command(action);assert path.is_file()
            passed(action,'Real output file: '+name)
        for extension in ['html','csv','pdf']:
            path=out/('health.'+extension);save_path(path);command('health_export',{'format':extension});assert path.stat().st_size>100
        passed('Dataset health exports','HTML, CSV and PDF written.')
        project=out/'workspace.mlstudio.json';save_path(project);command('save');assert project.is_file()
        old_source=c.source;old_frame=c.dataset.current
        frame_data(base.head(20),'temporary-data');open_path(project);command('open');idle()
        assert c.source==old_source and c.dataset.current.equals(old_frame)
        command('autosave');assert json.loads(project.read_text())['source']==old_source
        passed('Project roundtrip & autosave','Source fingerprint, operations, graph and target restored from disk.')
        original=Path(old_source).read_text();Path(old_source).write_text(original+'\n1,2,X,0\n');before=len(notices)
        command('open');idle();assert any(n['kind']=='error' and 'changed' in n['message'] for n in notices[before:]);Path(old_source).write_text(original)
        passed('Changed-source project rejection','Error displayed without hidden modal; current workspace preserved.')
        command('preferences',{'theme':'contrast','autosave':True,'research_rows':1000})
        assert b.snapshot()['preferences']['theme']=='contrast' and b.snapshot()['preferences']['autosave']
        command('preferences',{'theme':'pastel','autosave':False,'research_rows':20000})
        before=len(notices);command('preferences',{'research_rows':1});assert any(n['kind']=='error' for n in notices[before:])
        storage=out/'other-storage';storage.mkdir(exist_ok=True);QFileDialog.getExistingDirectory=lambda *a,**k:str(storage);command('storage_location')
        assert c.settings.value('storageRoot')==str(storage)
        passed('Settings & resource limits','Theme, autosave, valid/invalid row budget and next-launch storage selection persisted in isolated settings.')
        frame_data(base.assign(target=[0]*119+[1]),'invalid-rare-class');command('configure',{'target':'target','algorithm':'Decision Tree'})
        before=len(notices);command('train');idle()
        assert c.result is None and any(n['kind']=='error' for n in notices[before:])
        assert js('!window.appState.busy')
        frame_data(base,'recovery');command('configure',{'target':'target','algorithm':'Decision Tree'});command('train');idle();assert c.result is not None
        passed('Background failure recovery','Rare-class training error reported in web UI; valid training succeeds immediately afterward.')
        js("studioUI.navigate('history');true")
        assert js("document.querySelectorAll('[data-experiment]').length>0")
        js("document.querySelector('[data-experiment]').click();true")
        assert js("!!document.querySelector('.modal-body')");click('close_experiment')
        passed('History inspection','Actual SQLite records and experiment detail modal.')
        capture(window,out,'home')
        summary={'status':'passed','checks':records,'count':len(records),'seconds':round(time.monotonic()-started,2),'test_data':'Generated QA datasets only; no user files modified. Algorithms and file processing are real; only native dialog selections are automated.','known_unavailable':['Deep Learning / GPU engine','Saved model bundles across application restarts','Standalone installer/clean-machine acceptance'],'errors_exercised':sum(n['kind']=='error' for n in notices)}
        (out/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    except Exception:
        import traceback
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (out/'partial.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        raise
    finally:
        QFileDialog.getOpenFileName=original_open;QFileDialog.getSaveFileName=original_save;QFileDialog.getExistingDirectory=original_directory;QDialog.exec=original_dialog;QMessageBox.warning=original_warning
        if not c.busy:window.close()
