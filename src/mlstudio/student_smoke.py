"""Check the student presentation keeps feature controls and real workflows reachable."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings,QEventLoop,QTimer
from PySide6.QtWidgets import QApplication
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/student-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel')
    w.resize(1500,960);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    def js(code):return evaluate(w,code)
    def pause(ms=150):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def inventory():
        return json.loads(js("JSON.stringify(Array.from(document.querySelectorAll('#content [id],#content [data-action],#content [data-config],#content [data-ref],#content [data-request-feature],#content [data-prep-operation],#content [data-prep-issue],#content [data-visual-export],#content [data-health-export],#content [data-tab],#content [data-visual-example]')).map(el=>Array.from(el.attributes).filter(a=>['id','data-action','data-config','data-ref','data-request-feature','data-prep-operation','data-prep-column','data-prep-issue','data-visual-export','data-visual-id','data-health-export','data-tab','data-visual-example'].includes(a.name)).map(a=>a.name+'='+a.value).sort().join('|')).sort())"))
    checks=[]
    assert js("document.body.classList.contains('student-workspace')")
    assert js("!document.querySelector('.topbar-actions #student-help')")
    assert js("document.querySelectorAll('#navigation [data-page]').length===7")
    frame=pd.DataFrame({'measurement':np.r_[np.linspace(2,5,119),100.],
                        'category':['A','B']*60,'score':np.linspace(10,100,120),'outcome':[0,1]*60})
    frame.loc[0,'measurement']=np.nan
    source=out/'synthetic-student.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    wait_for(w,'window.appState?.dataset?.rows===120')
    routes=['home','data','preprocessing','analytics','models','results','advanced','analyzer','safety','reliability','impact','errors','researchpack','workflow','history','predictions','reports','settings']
    inventories=[]
    for page in routes:
        js('studioUI.navigate('+json.dumps(page)+');studentRender();true')
        before=inventory()
        js('studentApply();true')
        after=inventory()
        from collections import Counter
        assert not Counter(before)-Counter(after),(page,'Existing feature control was lost',Counter(before)-Counter(after))
        inventories.append({'page':page,'existing_controls':len(before),'after_student_controls':len(after)})
        assert js("document.querySelector('#content').textContent.length>50")
    checks.append('All 18 pages retain every existing action/input identity after student presentation')
    # The primary training and question controls still execute actual data operations.
    js("studioUI.navigate('models');true")
    js("document.getElementById('minimal-target').value='outcome';document.getElementById('minimal-target').dispatchEvent(new Event('change',{bubbles:true}));true")
    wait_for(w,"window.appState.target==='outcome'")
    js("document.querySelector('[data-action=simple_train]').click();true")
    wait_for(w,'!!window.appState.model&&!window.appState.busy',timeout=60)
    assert c.result.metadata['training_rows']==90
    checks.append('Friendly Train button runs a real model and displays actual holdout metrics')
    js("studioUI.navigate('analytics');document.querySelector('[data-visual-example=\"0\"]').click();document.querySelector('[data-action=visual_prepare]').click();true")
    wait_for(w,'!!window.appState.visual_requests.review')
    js("document.querySelector('[data-action=visual_execute]').click();true")
    wait_for(w,'window.appState.visual_requests.results.length===1')
    checks.append('Example question -> reviewable plan -> calculated visual result')
    js("studioUI.navigate('preprocessing');true")
    issue=next(i for i in c.issues if i.title=='Potential outliers' and i.columns==['measurement'])
    js("document.querySelector('[data-prep-issue=\""+issue.id+"\"]').click();true")
    wait_for(w,"window.appState.preprocessing.detail?.id==='"+issue.id+"'")
    assert js("document.querySelector('.prep-review .prep-evidence-table').textContent.includes('100')")
    assert js("document.querySelector('.prep-review .prep-fences')!==null")
    js("document.querySelector('[data-prep-operation=\"Cap outliers (IQR)\"]').click();true")
    wait_for(w,'!!window.appState.preprocessing.preview')
    assert js("!!document.querySelector('[data-action=preprocess_apply]')")
    assert c.dataset.index==0
    checks.append('Unusual-values card keeps exact records, expandable limits, caution and before/after controls')
    for value,label in [('pastel','light'),('dark','dark')]:
        folder=out/label;folder.mkdir(exist_ok=True)
        js("studioUI.command('preferences',{theme:"+json.dumps(value)+"});true")
        wait_for(w,"document.documentElement.dataset.theme==="+json.dumps(value))
        for page in ['home','data','preprocessing','analytics','models','advanced']:
            capture(w,folder,page)
            assert js("document.querySelectorAll('#navigation [data-page]').length===7")
        # Representative normal text remains legible in both palettes.
        ratios=json.loads(js("JSON.stringify((()=>{const rgb=s=>s.match(/[\\d.]+/g).slice(0,3).map(Number),lum=c=>c.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0),ratio=(a,b)=>(Math.max(a,b)+.05)/(Math.min(a,b)+.05);const surface=document.querySelector('.panel');return ['.panel-head h2','.panel-head p','.minimal-hub-card small'].map(sel=>{const node=document.querySelector(sel);return node?{selector:sel,ratio:ratio(lum(rgb(getComputedStyle(node).color)),lum(rgb(getComputedStyle(surface).backgroundColor)))}:null}).filter(Boolean)})())"))
        assert ratios and all(r['ratio']>=4.5 for r in ratios),(value,ratios)
    checks.append('Light/Dark page screenshots and representative text contrast >=4.5:1')
    # A user-expanded explanation stays open through status refreshes.
    js("studioUI.navigate('models');document.querySelector('#minimal-advanced').open=true;true")
    pause();b.publish();pause()
    assert js("document.querySelector('#minimal-advanced').open")
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'checks':checks,'control_inventory':inventories},indent=2),encoding='utf-8')
    w.close();app.processEvents()


if __name__=='__main__':run()
