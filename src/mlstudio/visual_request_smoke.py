"""Native request-to-result acceptance with the project's embedded WebEngine."""
import json
from pathlib import Path
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings,QEventLoop,QTimer
from sklearn.datasets import load_iris
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/visual-requests-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    window=ModernWindow(out/'storage');c,b=window.controller,window.bridge
    c.settings=QSettings(str(out/'appearance.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel')
    window.resize(1500,1000);window.show()
    wait_for(window,"document.body.dataset.ready==='true'")
    iris=load_iris(as_frame=True);frame=iris.data.copy()
    frame.columns=['sepal_length','sepal_width','petal_length','petal_width']
    frame['species']=[iris.target_names[i] for i in iris.target]
    source=out/'iris-validation.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    wait_for(window,'window.appState?.dataset?.rows===150')
    def js(code):return evaluate(window,code)
    js("studioUI.navigate('analytics');true")
    assert js("!!document.querySelector('.requirement-studio')")
    checks=[]
    def prepare(question):
        js("document.getElementById('visual-question').value="+json.dumps(question)+";document.querySelector('[data-action=visual_prepare]').click();true")
        wait_for(window,"window.appState.visual_requests.review?.question==="+json.dumps(question))
        assert js("!!document.querySelector('.visual-plan')")
    def execute():
        prior=b.visual_results[0]['id'] if b.visual_results else ''
        js("document.querySelector('[data-action=visual_execute]').click();true")
        wait_for(window,"(window.appState.visual_requests.results[0]?.id||'')!=="+json.dumps(prior))
        assert js("document.querySelector('.visual-result .chart-image')?.complete")
    # Exercise question chips: the visible field must agree with its in-memory draft.
    js("document.querySelector('[data-visual-example=\"0\"]').click();true")
    assert js("document.getElementById('visual-question').value.startsWith('Show average')")
    prepare('Show average sepal_length by species')
    assert b.visual_review['ready'];execute()
    assert abs(b.visual_results[0]['table']['data'][0][1]-5.006)<1e-9
    checks.append('Written question maps to actual Iris group means')
    assert b.visual_results[0]['support']['data'][0][1:]==[50,50]
    prepare('Count the number of flowers for each species in a bar chart');execute()
    assert [r[1] for r in b.visual_results[0]['table']['data']]==[50,50,50]
    checks.append('Count includes all 150 rows')
    prepare('Average all numeric features by species in separate vertical subplots');execute()
    assert len(b.visual_figures[b.visual_results[0]['id']].axes)==3
    checks.append('Four measurements in three species subplots')
    prepare('Scatter sepal_width versus sepal_length');execute()
    assert b.visual_results[0]['plan']['columns']==['sepal_width','sepal_length']
    checks.append('Scatter follows requested X/Y order')
    prepare('Show a histogram of sepal_length with 12 bins')
    # Explicitly edit filters and labels through the controls, then calculate.
    js("document.querySelector('.visual-plan details').open=true;document.querySelector('#visual-filter_column').value='species';document.querySelector('#visual-filter_column').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#visual-filter_operator').value='eq';document.querySelector('#visual-filter_operator').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#visual-filter_value').value='setosa';document.querySelector('#visual-filter_value').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#visual-title').value='Setosa sepal distribution';document.querySelector('#visual-title').dispatchEvent(new Event('input',{bubbles:true}));true")
    execute();assert b.visual_results[0]['rows_used']==50
    assert b.visual_results[0]['title']=='Setosa sepal distribution'
    assert sum(r[2] for r in b.visual_results[0]['table']['data'])==50
    checks.append('Edited row filter/title produce 50-row histogram')
    prepare('Make a pie chart of species counts')
    assert not b.visual_review['ready'] and b.visual_review['questions']
    checks.append('Unsupported request asks for choices')
    # Preserve input drafts and editable plans across status publication.
    js("document.getElementById('visual-question').value='Unsaved question draft';document.getElementById('visual-question').dispatchEvent(new Event('input',{bubbles:true}));true")
    b.publish();wait_for(window,"document.getElementById('visual-question').value==='Unsaved question draft'")
    checks.append('Draft text survives state refresh')
    prepare('Show a histogram of sepal_length with 12 bins')
    capture(window,out,'analytics');(out/'analytics.png').replace(out/'light-requirement.png')
    js("studioUI.command('preferences',{theme:'dark'});true")
    wait_for(window,"document.documentElement.dataset.theme==='dark'")
    capture(window,out,'analytics');(out/'analytics.png').replace(out/'dark-requirement.png')
    # Screenshot the real result, rather than just its controls.
    js("document.querySelector('.visual-result').scrollIntoView({block:'start'});true")
    loop=QEventLoop();QTimer.singleShot(750,loop.quit);loop.exec()
    window.grab().save(str(out/'dark-result.png'))
    b.visual_figures[b.visual_results[0]['id']].savefig(out/'filtered-histogram.png',dpi=110,bbox_inches='tight',facecolor='#1a2133')
    old_dialog=QFileDialog.getSaveFileName
    try:
        for format in ['csv','html','json','png','svg']:
            target=out/('visual-summary.'+format)
            QFileDialog.getSaveFileName=lambda *a,p=target:(str(p),'')
            b.command('visual_export',json.dumps({'id':b.visual_results[0]['id'],'format':format}))
            assert target.is_file() and target.stat().st_size>50
    finally:QFileDialog.getSaveFileName=old_dialog
    checks.append('Real CSV/HTML/JSON/PNG/SVG exports')
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
    window.close();app.processEvents()


if __name__=='__main__':run()
