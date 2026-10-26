"""Native acceptance of all three lab workflows using sklearn's local Iris data."""
import json
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from sklearn.datasets import load_iris
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/lab-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    window=ModernWindow(out/'storage')
    c,b=window.controller,window.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    window.resize(1500,1000);window.show()
    wait_for(window,"document.body.dataset.ready==='true'")
    assert evaluate(window,"document.querySelector('#remove-dataset')?.disabled===true")
    iris=load_iris(as_frame=True)
    df=iris.data.copy();df.columns=['sepal_length','sepal_width','petal_length','petal_width']
    df['species']=[iris.target_names[i] for i in iris.target]
    source=out/'iris-acceptance.csv';df.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(df);c.issues=analyze(df);c.refresh_data();b.publish()
    wait_for(window,'window.appState?.dataset?.rows===150')
    def js(script):return evaluate(window,script)
    js("studioUI.navigate('data');true")
    assert js("document.querySelector('.dataset-management [data-action=remove_dataset]')?.disabled===false")
    assert js("document.querySelector('#remove-dataset')?.disabled===false")
    assert js("document.querySelector('.dataset-insights').textContent.includes('Complete rows')")
    js("document.querySelector('#insight-row').value='2';document.querySelector('[data-action=inspect_row]').click();true")
    wait_for(window,"window.appState.row_detail?.position===2")
    assert b.row_detail['values'][0]['value']==4.9
    js("document.querySelector('#insight-column').value='species';document.querySelector('#insight-column').dispatchEvent(new Event('change',{bubbles:true}));true")
    assert js("document.querySelector('#insight-column').value==='species'")
    assert js("document.querySelector('.dataset-insights').textContent.includes('setosa')")
    capture(window,out,'data')
    js("studioUI.navigate('analytics');true")
    assert js("!!document.querySelector('#lab-studio')")
    for action in ['lab_mean','lab_count','lab_subplots']:
        js(f"document.querySelector('[data-action={action}]').click();document.querySelector('[data-action=lab_run]').click();true")
        expected='count' if action=='lab_count' else 'mean'
        wait_for(window,"!!window.appState.lab")
        wait_for(window,"document.querySelectorAll('#lab-studio .chart-image').length===1")
        if action=='lab_count':assert [r[1] for r in b.lab_result['table']['data']]==[50,50,50]
        elif action=='lab_mean':assert abs(b.lab_result['table']['data'][0][1]-5.006)<1e-9
        else:
            assert len(b.lab_figure.axes)==3
            assert len(b.lab_figure.axes[0].patches)==4
        capture(window,out,'analytics')
        (out/'analytics.png').replace(out/(action+'.png'))
    js("studioUI.command('preferences',{theme:'dark'});true")
    wait_for(window,"document.documentElement.dataset.theme==='dark'")
    capture(window,out,'analytics')
    assert js("document.querySelector('#lab-layout').value==='subplots'")
    assert js("document.querySelectorAll('#lab-studio input[data-lab-feature]:checked').length===4")
    b.lab_figure.savefig(out/'three-subplots.png',dpi=110,bbox_inches='tight',facecolor=b.lab_figure.get_facecolor())
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':['Mean by group: actual Iris mean 5.006','Counts: 50 per species','Three vertical subplots with four measurements','Light/dark UI and chart','Preset controls and QWebChannel']},indent=2),encoding='utf-8')
    window.close();app.processEvents()


if __name__=='__main__':run()
