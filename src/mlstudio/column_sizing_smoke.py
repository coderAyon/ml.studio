"""Verify real header dragging and wide-table width controls in embedded WebEngine."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint,Qt,QSettings
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/column-sizing-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1700,1050);w.show();js=lambda s:evaluate(w,s)
    def point(selector):
        p=json.loads(js("JSON.stringify((()=>{const r=document.querySelector("+json.dumps(selector)+").getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})())"))
        return QPoint(round(p['x']),round(p['y']))
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        f=pd.DataFrame({f'feature_{i}':np.arange(245)+i for i in range(30)});f['long_description']=['A much longer description that needs a wider custom column']*245
        original=f.copy(deep=True);source=out/'synthetic-wide.csv';f.to_csv(source,index=False)
        c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();b.publish();wait_for(w,'state.dataset?.rows===245')
        js("studioUI.navigate('data');document.querySelector('.easy-dataset-table .table-scroll').scrollIntoView({block:'start'});window.scrollBy(0,-110);true");QTest.qWait(450)
        before=js("document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width")
        surface=w.view.focusProxy() or w.view;start=point('[data-column-grip=feature_0]');end=start+QPoint(90,0)
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(surface,end,160);QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end);QTest.qWait(100)
        after=js("document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width")
        assert after>before+75 and after<before+105,(before,after)
        assert js("explorer.sort===''&&!document.body.classList.contains('column-resizing')")
        js("explorer.sort='feature_0';explorer.descending=true;explorer.page=1;queryExplorer();true")
        wait_for(w,'state.dataset.table_page===1')
        assert abs(js("document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width")-after)<1
        js("explorer.hidden.add('feature_0');render();explorer.hidden.delete('feature_0');render();true")
        assert abs(js("document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width")-after)<1
        js("document.getElementById('column-size-settings').open=true;document.getElementById('column-size-value').value='310';document.querySelector('[data-column-size=apply]').click();true")
        assert js("Math.round(document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width)===310")
        js("document.querySelector('[data-column-size=compact]').click();true")
        assert js("Math.round(document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width)===88")
        js("document.querySelector('[data-column-size=fit]').click();true")
        assert js("columnWidth('long_description')>300")
        js("document.querySelector('[data-column-size=reset]').click();true")
        assert js("Math.round(document.querySelector('th[data-width-column=feature_0]').getBoundingClientRect().width)===160")
        js("document.querySelector('.scalable-table').closest('.table-scroll').scrollLeft=400;render();true")
        assert js("document.querySelector('.scalable-table').closest('.table-scroll').scrollLeft===400")
        js("document.querySelector('.scalable-table').closest('.table-scroll').scrollLeft=0;document.getElementById('column-size-value').value='240';document.querySelector('[data-column-size=apply]').click();document.querySelector('#toast').classList.remove('show');true")
        js("document.getElementById('workspace-table-options')?.setAttribute('open','');document.getElementById('column-size-settings').open=true;document.querySelector('.scalable-table').closest('.table-scroll').scrollTop=0;document.getElementById('column-size-settings').scrollIntoView({block:'start'});window.scrollBy(0,-90);true");w.raise_();w.activateWindow();QTest.qWait(650);w.grab().save(str(out/'custom-width-light.png'))
        js("cmd('preferences',{theme:'dark'});true");wait_for(w,"state.preferences.theme==='dark'&&document.documentElement.dataset.theme==='dark'");QTest.qWait(700)
        w.grab().save(str(out/'custom-width-dark.png'))
        pd.testing.assert_frame_equal(c.dataset.current,original)
        (out/'result.json').write_text(json.dumps({'passed':True,'columns':31,'rows':245,'native_drag_before':before,'native_drag_after':after,
            'checks':['Native header drag adjusts rendered width','Resize does not sort column','Width survives sort/page/hide/show',
                'Exact numeric width','Compact preset','Auto-fit long values','Reset widths','Horizontal scroll survives render',
                'Light/Dark screenshots','Original data unchanged']},indent=2));print('Column sizing desktop smoke passed')
    finally:w.close();app.processEvents()


if __name__=='__main__':run()
