"""Native acceptance of reference dashboard and dataset controls, synthetic data."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QSettings
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate, wait_for, capture


def run(window, application, directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    window.controller.settings=QSettings(str(directory/"acceptance-settings.ini"),QSettings.Format.IniFormat)
    window.resize(1680,980);window.show()
    wait_for(window,"document.body.dataset.ready === 'true'")
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({"measurement":rng.normal(5,1,245),"age":rng.integers(18,80,245),"category":[" Group A","Group B","Group C","Group A","Group B"]*49,"outcome":[0,1,0,1,0]*49})
    frame.loc[3:8,"measurement"]=np.nan
    frame.loc[244,"category"]="unique last record"
    source=directory/"synthetic-pastel.csv";frame.to_csv(source,index=False)
    c=window.controller;c.source=str(source);c.import_options={"delimiter":",","encoding":"utf-8-sig","header":True,"decimal":"."}
    c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();window.bridge.publish()
    wait_for(window,"window.appState?.dataset?.rows === 245")
    assert evaluate(window,"document.querySelector('.hero').textContent.includes('Welcome back')")
    assert evaluate(window,"document.querySelectorAll('.automatic-chart svg').length === 0")
    evaluate(window,"studioUI.navigate('analytics');true")
    assert evaluate(window,"document.querySelectorAll('.automatic-chart svg').length === 3")
    evaluate(window,"studioUI.navigate('home');true")
    capture(window,directory,"home")
    evaluate(window,"studioUI.navigate('data');document.getElementById('dataset-search').value='unique last record';document.getElementById('dataset-search').dispatchEvent(new Event('input',{bubbles:true}));true")
    wait_for(window,"window.appState.dataset.filtered_rows===1")
    assert evaluate(window,"document.querySelector('tbody').textContent.includes('unique last record')")
    evaluate(window,"view.search='';explorer.page=1;explorer.sort='age';queryExplorer();true")
    wait_for(window,"window.appState.dataset.table_page===1")
    assert evaluate(window,"document.querySelectorAll('tbody tr').length===100")
    evaluate(window,"document.querySelector('[data-visible=category]').click();true")
    assert not evaluate(window,"Array.from(document.querySelectorAll('thead th')).some(e=>e.textContent==='category')")
    evaluate(window,"document.querySelector('[data-visible=category]').click();document.querySelector('[data-select-row]').click();true")
    assert evaluate(window,"explorer.selected.size===1")
    capture(window,directory,"data")
    evaluate(window,"view.dataTab='schema';explorer.column='measurement';render();true")
    assert evaluate(window,"document.querySelector('.profile-stats').textContent.includes('Standard deviation')")
    capture(window,directory,"data")
    window.grab().save(str(directory/"column-profile.png"))
    evaluate(window,"document.getElementById('clean-operation').value='Rename column';document.getElementById('clean-column').value='age';document.getElementById('clean-value').value='patient_age';document.querySelector('[data-action=preview]').click();true")
    wait_for(window,"!!window.appState.preview")
    evaluate(window,"document.querySelector('[data-action=apply_preview]').click();true")
    wait_for(window,"window.appState.dataset.version===1&&!window.appState.busy")
    assert "patient_age" in c.dataset.current
    evaluate(window,"studioUI.command('undo');true")
    wait_for(window,"window.appState.dataset.version===0&&!window.appState.busy")
    evaluate(window,"studioUI.navigate('analyzer');true")
    assert evaluate(window,"document.querySelector('.health-formula').textContent.includes('Completeness')")
    choices=evaluate(window,"document.querySelectorAll('[data-plan-index]').length")
    assert choices>0
    evaluate(window,"document.querySelector('[data-plan-index]').checked=true;document.querySelector('[data-ref=bulk]').click();true")
    wait_for(window,"!!window.appState.preview?.operations")
    evaluate(window,"document.querySelector('[data-action=cancel_preview]').click();true")
    capture(window,directory,"analyzer")
    original=QFileDialog.getSaveFileName
    try:
        for extension in ['html','csv','pdf']:
            path=directory/f"dataset-health.{extension}"
            QFileDialog.getSaveFileName=lambda *a,p=path,**k:(str(p),"")
            evaluate(window,f"document.querySelector('[data-health-export={extension}]').click();true")
            assert path.is_file() and path.stat().st_size>100
    finally:QFileDialog.getSaveFileName=original
    evaluate(window,"studioUI.navigate('workflow');document.querySelector('[data-action=starter]').click();true")
    wait_for(window,"Object.keys(window.appState.workflow.nodes).length===3")
    evaluate(window,"document.querySelector('[data-ref=layout]').click();true")
    wait_for(window,"window.appState.workflow.nodes['2'].position[0]===270")
    assert evaluate(window,"!!document.getElementById('workflow-minimap')")
    evaluate(window,"document.getElementById('node-search').value='train';document.getElementById('node-search').dispatchEvent(new Event('input',{bubbles:true}));true")
    assert evaluate(window,"Array.from(document.querySelectorAll('.library-node')).filter(n=>!n.hidden).length===1")
    capture(window,directory,"workflow")
    # Native save boundary is the only mocked part; actual autosave writes project JSON.
    project=directory/"autosave.mlstudio.json"
    c.save_project(str(project))
    evaluate(window,"studioUI.command('autosave');true")
    assert json.loads(project.read_text())["source"]==str(source)
    capture(window,directory,"settings")
    (directory/"result.json").write_text(json.dumps({"status":"passed","fixture":"Synthetic 245-row dataset","checks":["Pastel dashboard; charts live in Visual Analytics","Full-dataset search including last row","Sorting, pagination, visibility and row selection","Actual column statistics","Rename preview, apply and undo","Health formula and multi-step cleaning-plan preview","HTML, CSV and PDF health exports","Workflow search, auto layout and minimap","Existing-project autosave"]},indent=2),encoding="utf-8")
    window.close()
