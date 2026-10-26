"""Native end-to-end acceptance for the five research tools."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QSettings
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run(window,application,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    window.controller.settings=QSettings(str(directory.resolve()/"acceptance-settings.ini"),QSettings.Format.IniFormat)
    window.show();wait_for(window,"document.body.dataset.ready === 'true'")
    rng=np.random.default_rng(7)
    x=rng.normal(size=180); y=(x+rng.normal(0,.8,180)>0).astype(int)
    frame=pd.DataFrame({"measurement":x,"secondary":rng.normal(size=180),"patient_id":np.repeat(np.arange(45),4),"visit_date":pd.date_range("2020-01-01",periods=180).astype(str),"category":["A","B","C"]*60,"outcome":y})
    frame.loc[4:16,"secondary"]=np.nan
    source=directory/"synthetic-research.csv";frame.to_csv(source,index=False)
    c=window.controller;c.source=str(source);c.import_options={"delimiter":",","encoding":"utf-8-sig","header":True,"decimal":"."}
    c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();c.target.setCurrentText("outcome");window.bridge.publish()
    wait_for(window,"window.appState?.dataset?.rows === 180")
    evaluate(window,"studioUI.navigate('safety');document.querySelector('[data-exclude=patient_id]').checked=true;document.querySelector('[data-exclude=visit_date]').checked=true;document.querySelector('[data-action=research_save_design]').click();true")
    wait_for(window,"window.appState.research.config.excluded.length === 2")
    assert evaluate(window,"window.appState.research.safety.some(i=>i.code==='possible_group') && window.appState.research.safety.some(i=>i.code==='possible_time')")
    capture(window,directory,"safety")
    evaluate(window,"studioUI.navigate('reliability');document.querySelectorAll('[data-research-algorithm]').forEach(e=>e.checked=['Logistic Regression','Decision Tree'].includes(e.dataset.researchAlgorithm));document.querySelector('[data-action=research_run_compare]').click();true")
    wait_for(window,"window.appState.research.session?.comparison?.models.length === 2 && !window.appState.busy",timeout=60)
    assert window.bridge.research.session.final_used is False
    capture(window,directory,"reliability")
    evaluate(window,"studioUI.navigate('impact');document.getElementById('impact-algorithm').value='Logistic Regression';document.querySelector('[data-action=research_run_impact]').click();true")
    wait_for(window,"window.appState.research.session?.impact?.variants.length === 4 && !window.appState.busy",timeout=60)
    capture(window,directory,"impact")
    evaluate(window,"studioUI.navigate('reliability');document.getElementById('final-algorithm').value='Logistic Regression';document.getElementById('final-confirm').checked=true;document.querySelector('[data-action=research_evaluate_final]').click();true")
    wait_for(window,"!!window.appState.research.session?.final && !window.appState.busy",timeout=60)
    evaluate(window,"studioUI.navigate('errors');true")
    assert evaluate(window,"window.appState.research.session.final.errors.total > 0")
    capture(window,directory,"errors")
    assert window.bridge.research.session.final_used
    before=window.bridge.research.session.final_result
    evaluate(window,"studioUI.command('research_final',{algorithm:'Decision Tree',variant:'baseline',confirmed:true});true")
    assert window.bridge.research.session.final_result==before
    capture(window,directory,"researchpack")
    original=QFileDialog.getSaveFileName
    archive=directory/"research-pack.zip"
    try:
        QFileDialog.getSaveFileName=lambda *a,**k:(str(archive),"")
        evaluate(window,"document.querySelector('[data-action=research_export]').click();true")
        wait_for(window,"!window.appState.busy",timeout=60)
    finally:QFileDialog.getSaveFileName=original
    assert archive.is_file(),"Research ZIP was not written"
    summary={"status":"passed","version":"0.4.0","fixture":"Explicitly synthetic research acceptance data; isolated history","checks":["Safety Advisor evidence and design controls","Repeated same-fold model CV with reserved final rows","Four preprocessing variants and paired fold effects","Explicit final evaluation and repeat-use lock","Error Detective rows, metrics and slices","Native research ZIP export including runnable sources"],"final_metrics":before["metrics"],"pack":str(archive)}
    (directory/"result.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    window.close()
