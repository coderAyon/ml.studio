"""Real WebEngine layout checks with isolated synthetic data and screenshots."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PySide6.QtCore import Qt, QSettings, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
from mlstudio.desktop import ModernWindow
from mlstudio.data import DatasetVersions
from mlstudio.analyzer import analyze
from mlstudio.ui_smoke import evaluate, wait_for
import pandas as pd
import numpy as np


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    w = ModernWindow(out / 'storage')
    c = w.controller
    c.settings = QSettings(str(out / 'settings.ini'), QSettings.Format.IniFormat)
    frame = pd.DataFrame({'measurement': np.arange(120, dtype=float), 'category': ['A','B']*60, 'label':[0,1]*60})
    frame.loc[3, 'measurement'] = np.nan
    source = out / 'synthetic.csv'
    frame.to_csv(source, index=False)
    original = source.read_bytes()
    c.source, c.dataset = str(source), DatasetVersions(frame)
    c.issues = analyze(frame)
    c.refresh_data()
    w.show()
    wait_for(w, "document.body.dataset.ready==='true'")
    report = []
    for theme in ['pastel','dark']:
        evaluate(w, f"cmd('preferences',{{theme:'{theme}',reduce_motion:false}});true")
        wait_for(w, f"document.documentElement.dataset.theme==='{theme}'")
        for width,height in [(1440,900),(1024,768),(800,640)]:
            w.resize(width,height)
            for page in ['data','dataset_visuals','preprocessing','models','workflow','results','advanced','settings']:
                evaluate(w, f"navigate('{page}');true")
                loop = QEventLoop();QTimer.singleShot(400,loop.quit);loop.exec()
                metrics = json.loads(evaluate(w, """JSON.stringify((()=>{const c=document.querySelector('#content'),s=getComputedStyle(c),b=document.querySelector('.button'),n=document.querySelector('.nav-item');return {width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth-innerWidth,contentWidth:c.clientWidth,bodyFont:getComputedStyle(document.body).fontFamily,buttonFont:b?getComputedStyle(b).fontSize:null,navFont:n?getComputedStyle(n).fontSize:null,panels:[...c.querySelectorAll('.panel')].length,animations:document.getAnimations().filter(a=>a.playState==='running').length}})())"""))
                metrics.update(page=page,theme=theme)
                report.append(metrics)
                if page in ['data','preprocessing','workflow'] and (width in [800,1440]):
                    w.grab().save(str(out/f'{theme}-{width}-{page}.png'))
    evaluate(w,"cmd('preferences',{reduce_motion:true});true")
    wait_for(w,"document.documentElement.dataset.motion==='reduced'")
    reduced = evaluate(w,"document.getAnimations().filter(a=>a.playState==='running').length")
    result = {'layouts':report,'max_overflow':max(r['overflow'] for r in report),'reduced_motion_running':reduced,'source_preserved':source.read_bytes()==original}
    assert {r['width'] for r in report} >= {800,1024,1440}, 'Window size clamped; responsive checks did not reach requested sizes'
    assert result['max_overflow'] <= 1 and reduced == 0 and result['source_preserved']
    assert all(float(r['buttonFont'][:-2]) >= 12 for r in report if r['buttonFont']), 'Unreadably small action text'
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'max_overflow':result['max_overflow'],'overflow_pages':[r for r in report if r['overflow']>1],'reduced_motion_running':reduced,'source_preserved':result['source_preserved']}))
    w.close();app.processEvents()


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
