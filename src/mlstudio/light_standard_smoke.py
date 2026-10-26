"""Visual acceptance of the stronger Light palette; no user preferences are changed."""
import json
from pathlib import Path
import pandas as pd
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run():
    out=Path('examples/light-standard-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'preferences.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('referenceTheme','pastel');w.resize(1530,980);w.show()
    wait_for(w,"document.body.dataset.ready==='true'")
    frame=pd.DataFrame({'length':[1.,2.,3.,4.]*30+[100.],'width':[2.,4.,6.,8.]*30+[200.],
                        'species':['A','B','C']*40+['A']})
    frame.loc[0,'length']=None
    source=out/'synthetic-light-qa.csv';frame.to_csv(source,index=False)
    c.source=str(source);c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();b.publish()
    wait_for(w,'window.appState?.dataset?.rows===121')
    def js(code):return evaluate(w,code)
    results=[]
    contrast="""(()=>{const rgb=s=>s.match(/[\\d.]+/g).slice(0,3).map(Number),lum=c=>c.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0),ratio=(a,b)=>(Math.max(a,b)+.05)/(Math.min(a,b)+.05);return [['navigation','#navigation .nav-item:not(.active)','255,255,255'],['active navigation','#navigation .nav-item.active','231,222,250'],['heading','.panel-head h2','246,248,251'],['supporting text','.panel-head p','246,248,251'],['finding label','.prep-finding small','246,248,251'],['button','.button:not(.primary):not(:disabled)','255,255,255']].map(([name,sel,bg])=>{const node=document.querySelector(sel);return node?{name,ratio:ratio(lum(rgb(getComputedStyle(node).color)),lum(rgb(bg)))}:null}).filter(Boolean)})()"""
    for page in ['home','data','preprocessing','models','dataset_visuals','workflow']:
        js('studioUI.navigate('+json.dumps(page)+');true')
        assert js("document.documentElement.dataset.theme==='pastel'")
        assert js("getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()==='#e9edf4'")
        rows=json.loads(js('JSON.stringify('+contrast+')'))
        assert all(r['ratio']>=4.5 for r in rows),(page,rows)
        results.extend([dict(r,page=page) for r in rows])
        capture(w,out,page)
    # Stronger light styling must not leak into Dark or High contrast light.
    for theme in ['dark','contrast']:
        js("studioUI.command('preferences',{theme:"+json.dumps(theme)+"});true")
        wait_for(w,"document.documentElement.dataset.theme==="+json.dumps(theme))
        assert js("getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()!=='#e9edf4'")
    assert c.dataset.current.equals(frame)
    (out/'result.json').write_text(json.dumps({'passed':True,'light_only':True,'minimum_measured_text_contrast':min(r['ratio'] for r in results),'measurements':results},indent=2),encoding='utf-8')
    w.close();app.processEvents()


if __name__=='__main__':run()
