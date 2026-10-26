"""Native light/dark, text contrast, motion and preference acceptance."""
import base64
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PySide6.QtCore import QSettings,QEventLoop,QTimer
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for,capture


def run(window,application,directory):
    out=Path(directory).resolve();out.mkdir(parents=True,exist_ok=True)
    c=window.controller;b=window.bridge;c.settings=QSettings(str(out/'appearance.ini'),QSettings.Format.IniFormat)
    def js(code):return evaluate(window,code)
    def pause(ms=350):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def theme(value):
        js('studioUI.command("preferences",'+json.dumps({'theme':value})+');true')
        wait_for(window,'document.documentElement.dataset.theme==='+json.dumps(value));pause()
    try:
        window.resize(1680,980);window.show();wait_for(window,"document.body.dataset.ready==='true'")
        rng=np.random.default_rng(5)
        frame=pd.DataFrame({'measurement':rng.normal(5,1,150),'spending':rng.uniform(1,100,150),'group':['Group A','Group B','Group C']*50,'outcome':[0,1]*75})
        frame.loc[3:8,'measurement']=np.nan
        source=out/'synthetic-appearance.csv';frame.to_csv(source,index=False)
        c.source=str(source);c.import_options={'delimiter':',','encoding':'utf-8-sig','header':True,'decimal':'.'};c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.refresh_data();c.target.setCurrentText('outcome');c.algorithm.setCurrentText('Decision Tree');b.publish()
        wait_for(window,'window.appState?.dataset?.rows===150')
        js('studioUI.command("train");true');wait_for(window,'!!window.appState.model&&!window.appState.busy')
        js('studioUI.command("research_compare",{algorithms:["Logistic Regression","Decision Tree"]});true');wait_for(window,'!!window.appState.research.session?.comparison&&!window.appState.busy',timeout=60)
        js('studioUI.command("chart",{type:"Box plot",x:"measurement",y:"spending"});true');wait_for(window,'!!window.appState.chart')
        routes=['home','workflow','data','analyzer','analytics','models','history','predictions','reports','settings','safety','impact','reliability','errors','researchpack']
        contrast=[]
        contrast_script="""(()=>{const rgb=s=>{s=s.trim();if(s.startsWith('#')){const h=s.slice(1);const n=parseInt(h.length===3?h.split('').map(c=>c+c).join(''):h,16);return [(n>>16)&255,(n>>8)&255,n&255];}return s.match(/[\\d.]+/g).slice(0,3).map(Number);},lum=a=>a.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0),ratio=(a,b)=>{const x=lum(a),y=lum(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05)};const pairs=[['h2','.panel-head h2','--surface'],['body','body','--bg'],['navigation','.nav-item:not(:disabled)','--surface'],['active navigation','.nav-item.active','--purple-light'],['small supporting text','.stat-foot','--surface-soft'],['helper text','.hint','--surface'],['table values','td:not(.index)','--surface'],['table headings','th','--surface-soft']];return pairs.map(([name,selector,bg])=>{const node=document.querySelector(selector);if(!node)return null;return {name,ratio:ratio(rgb(getComputedStyle(node).color),rgb(getComputedStyle(document.documentElement).getPropertyValue(bg)))};}).filter(Boolean);})()"""
        for value,label in [('pastel','light'),('dark','dark')]:
            theme(value);folder=out/label;folder.mkdir(exist_ok=True)
            for page in routes:
                js('studioUI.navigate('+json.dumps(page)+');true')
                assert js('document.documentElement.dataset.theme==='+json.dumps(value))
                assert js("document.querySelector('#content').textContent.trim().length>20"),page
                if page in ['home','data','settings','workflow','analytics','analyzer','models','reliability']:
                    capture(window,folder,page)
                raw=js("JSON.stringify("+contrast_script+")")
                assert isinstance(raw,str),(label,page,raw)
                measurements=json.loads(raw)
                assert measurements,(label,page,"Contrast measurement produced no values")
                for row in measurements:
                    row.update(theme=label,page=page);contrast.append(row)
                    assert row['ratio']>=4.5,(label,page,row)
            svg=base64.b64decode(b.chart.split(',',1)[1]).decode()
            assert ('#1a2133' if value=='dark' else '#ffffff') in svg
            js("studioUI.navigate('home');document.querySelector('[data-action=quality_details]').click();true")
            pause(400);window.raise_();window.activateWindow();pause(250);window.grab().save(str(folder/'quality-popup.png'))
            assert js("!!document.querySelector('.quality-modal')")
            js("document.querySelector('[data-action=close_quality]').click();true")
        # Header button uses the same persisted preferences as Settings.
        js("document.getElementById('theme-toggle').click();true")
        wait_for(window,"window.appState.preferences.theme==='pastel'")
        js("document.getElementById('theme-toggle').click();true")
        wait_for(window,"window.appState.preferences.theme==='dark'")
        c.settings.sync();reopened=QSettings(str(out/'appearance.ini'),QSettings.Format.IniFormat)
        assert reopened.value('referenceTheme')=='dark'
        js("studioUI.navigate('data');document.getElementById('clean-operation').value='Rename column';document.getElementById('clean-value').value='unsaved draft';document.getElementById('dataset-search').focus();document.getElementById('dataset-search').setSelectionRange(0,0);true")
        b.publish();pause()
        assert js("document.activeElement.id==='dataset-search'&&document.getElementById('clean-value').value==='unsaved draft'&&document.getElementById('clean-operation').value==='Rename column'")
        pause(700);assert js("document.getElementById('content').getAnimations().filter(a=>a.playState==='running').length===0")
        b.on_status('Appearance acceptance: status refresh');pause(100)
        assert js("document.getElementById('content').getAnimations().filter(a=>a.playState==='running').length===0")
        os_reduced=js('appearance.media.matches')
        count=js("studioUI.navigate('analytics');document.getElementById('content').getAnimations().filter(a=>a.playState==='running').length")
        assert count==(0 if os_reduced else 1),(os_reduced,count)
        js("studioUI.navigate('settings');document.getElementById('pref-reduce-motion').click();true")
        wait_for(window,"window.appState.preferences.reduce_motion===true&&document.documentElement.dataset.motion==='reduced'")
        assert js("studioUI.navigate('data');document.getElementById('content').getAnimations().filter(a=>a.playState==='running').length===0")
        assert js("getComputedStyle(document.querySelector('.nav-item')).transitionDuration==='0s'")
        theme('contrast');assert js("document.documentElement.dataset.theme==='contrast'")
        c.settings.sync()
        (out/'result.json').write_text(json.dumps({'status':'passed','checks':['All 15 available pages in Light and Dark','Representative normal-size text contrast at least 4.5:1','Dark/light Matplotlib plotting paper and labels','Quality issue popup in both themes','Header theme toggle and persisted settings','Unsaved input, selection and keyboard-focus preservation','No entry-animation replay on status refresh','Page-entry motion when allowed','OS and saved reduced-motion handling','Legacy high-contrast light preference'],'minimum_measured_contrast':min(r['ratio'] for r in contrast),'contrast_measurements':contrast,'os_reduced_motion':os_reduced},indent=2),encoding='utf-8')
    except Exception:
        import traceback
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if not c.busy:window.close()
