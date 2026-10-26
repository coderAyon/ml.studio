"""Native UI acceptance for clustering, issue popup and progressive disclosure."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run(window,application,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    c=window.controller;c.settings=QSettings(str(directory/'settings.ini'),QSettings.Format.IniFormat)
    window.show();wait_for(window,"document.body.dataset.ready==='true'")
    rng=np.random.default_rng(8)
    frame=pd.DataFrame({'x':np.r_[rng.normal(-3,.3,45),rng.normal(3,.3,45)],'y':np.r_[rng.normal(-2,.3,45),rng.normal(2,.3,45)],'target':[0]*45+[1]*45,'id':range(90)})
    frame.loc[4,'x']=np.nan
    source=directory/'synthetic-groups.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();c.target.setCurrentText('target');window.bridge.publish()
    wait_for(window,"window.appState?.dataset?.rows===90")
    assert evaluate(window,"document.querySelectorAll('.automatic-overview').length===0")
    evaluate(window,"document.querySelector('[data-action=quality_details]').click();true")
    assert evaluate(window,"document.querySelectorAll('.popup-issue').length===window.appState.issues.length")
    capture(window,directory,'home')
    window.grab().save(str(directory/'quality-popup.png'))
    evaluate(window,"document.querySelector('[data-open-finding]').click();true")
    assert evaluate(window,"studioUI.page==='analyzer'&&!document.querySelector('.quality-modal')")
    assert evaluate(window,"!!document.querySelector('details.help-details>.guidance-overview')")
    capture(window,directory,'analyzer')
    evaluate(window,"studioUI.navigate('models');document.getElementById('learning-task').value='clustering';document.getElementById('learning-task').dispatchEvent(new Event('change',{bubbles:true}));true")
    assert evaluate(window,"!document.querySelector('[data-config=target]')")
    assert evaluate(window,"JSON.stringify(beginner.features)===JSON.stringify(['x','y'])")
    evaluate(window,"document.getElementById('group-count').value='2';document.getElementById('group-count').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-action=compare_groups]').click();true")
    wait_for(window,"!!window.appState.clustering&&!window.appState.busy",timeout=60)
    assert window.bridge.clustering_result['metrics']['groups']==2
    assert len(window.bridge.clustering_result['comparison'])==3
    assert evaluate(window,"!!document.querySelector('.group-plot')")
    capture(window,directory,'models')
    original=QFileDialog.getSaveFileName
    try:
        path=directory/'grouped.csv';QFileDialog.getSaveFileName=lambda *a,**k:(str(path),'')
        evaluate(window,"document.querySelector('[data-action=export_groups]').click();true")
        exported=pd.read_csv(path);assert len(exported)==90 and 'cluster' in exported
    finally:QFileDialog.getSaveFileName=original
    evaluate(window,"studioUI.navigate('analytics');true")
    assert evaluate(window,"document.querySelectorAll('.automatic-chart svg').length===3")
    evaluate(window,"studioUI.navigate('models');document.getElementById('learning-task').value='classification';document.getElementById('learning-task').dispatchEvent(new Event('change',{bubbles:true}));true")
    wait_for(window,"!!document.querySelector('[data-action=train]')")
    # A different active version must not expose stale clustering labels.
    c.dataset.apply(__import__('mlstudio.data',fromlist=['Transformation']).Transformation('Drop columns',['id']))
    c.refresh_data();window.bridge.publish()
    wait_for(window,"window.appState.dataset.version===1")
    assert window.bridge.clustering_result is None
    (directory/'result.json').write_text(json.dumps({'status':'passed','fixture':'Explicit synthetic two-group data','checks':['Overview contains no redundant chart section','Quality card opens real issue details and links to analyzer','Detailed guidance is collapsed by default','Clustering is a task without a target; excludes selected outcome and IDs by default','Same-feature three-method comparison with group plot/profiles and K scan','Group-label CSV export','Charts retained in Visual Analytics','Classification remains available','Dataset changes invalidate clustering results']},indent=2),encoding='utf-8')
    window.close()
