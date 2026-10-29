"""Native mode/report acceptance using isolated, clearly synthetic business records."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from PySide6.QtCore import Qt,QSettings,QEventLoop,QTimer
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
from mlstudio.desktop import ModernWindow
from mlstudio.data import DatasetVersions
from mlstudio.ui_smoke import evaluate,wait_for
import numpy as np
import pandas as pd


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat)
    c.settings.setValue('businessMode',False);c.settings.setValue('referenceTheme','pastel')
    f=pd.DataFrame({'sales':[100.,200.,50.,75.]*20,'cost':[60.,120.,30.,40.]*20,
        'region':['East','West','East','North']*20,'product':['A','B']*40,
        'date':pd.date_range('2026-01-01',periods=80).astype(str),'customer':['x','y','z','x']*20})
    source=out/'synthetic-business.csv';f.to_csv(source,index=False);original=source.read_bytes()
    c.source=str(source);c.import_options={};c.dataset=DatasetVersions(f);c.refresh_data();w.resize(1440,900);w.show()
    js=lambda code:evaluate(w,code)
    wait_for(w,"document.body.dataset.ready==='true'")
    assert js("document.querySelector('.sidebar-dataset strong').textContent==='synthetic-business.csv'")
    assert js("!!document.querySelector('.sidebar-dataset #remove-dataset')&&!document.querySelector('.topbar #remove-dataset')")
    js("document.querySelector('[data-workspace-mode=business]').click();true")
    wait_for(w,"state.business.enabled&&!!document.querySelector('#biz-measure')")
    cfg={'measure':'sales','aggregate':'sum','dimension':'region','pivot':'product','date':'date',
         'grain':'month','cost':'cost','customer':'customer','target':10000,'scenario_percent':10}
    # Exercise the real form controls/change handlers.
    for key,value in cfg.items():
        if key in {'target','scenario_percent'}:continue
        js(f"document.querySelector('#biz-{key}').value={json.dumps(value)};document.querySelector('#biz-{key}').dispatchEvent(new Event('change',{{bubbles:true}}));true")
    js("document.querySelector('[data-business-build]').click();true")
    wait_for(w,"!!state.business.result&&!state.busy")
    assert b.business.result['metric']['value']==8500
    js("navigate('business');true")
    assert js("document.querySelectorAll('.business-kpi').length===4")
    loop=QEventLoop();QTimer.singleShot(400,loop.quit);loop.exec()
    w.grab().save(str(out/'overview-light.png'))
    js("navigate('business_segments');Array.from(document.querySelectorAll('[data-business-segment]')).find(n=>n.textContent.includes('East')).click();true")
    wait_for(w,"state.business.result?.filtered_rows===40&&!state.busy")
    assert b.business.result['metric']['value']==3000
    js("document.querySelector('[data-business-clear-filters]').click();true")
    wait_for(w,"state.business.result?.filtered_rows===80&&!state.busy")
    for page in ['business_trends','business_pivot','business_data','business_report','business_scenario']:
        js(f"navigate('{page}');true");assert js("document.querySelector('#content').textContent.length>100")
    for key,value in [('target',10000),('scenario_percent',10)]:
        js(f"document.querySelector('#biz-{key}').value='{value}';document.querySelector('#biz-{key}').dispatchEvent(new Event('change',{{bubbles:true}}));true")
    js("document.querySelector('[data-business-build]').click();true")
    wait_for(w,"state.business.result?.scenario.percent===10&&!state.busy")
    assert b.business.result['scenario']['value']==9350
    old_save,old_open=QFileDialog.getSaveFileName,QFileDialog.getOpenFileName
    try:
        for kind in ['html','summary','segments','trend','pivot','data']:
            path=out/('business-'+kind+('.html' if kind=='html' else '.csv'))
            QFileDialog.getSaveFileName=lambda *args,p=path:(str(p),'')
            b.business.command('business_export',{'kind':kind});assert path.stat().st_size>10
        definition=out/'report-definition.json';QFileDialog.getSaveFileName=lambda *args:(str(definition),'')
        js("cmd('business_definition_save');true");assert definition.is_file()
        QFileDialog.getOpenFileName=lambda *args:(str(definition),'')
        js("cmd('business_definition_open');true");wait_for(w,"!state.busy")
        project=out/'business-project.json';c.save_project(str(project))
        assert json.loads(project.read_text())['business_config']['measure']=='sales'
        prior=c.dataset.revision_id;QFileDialog.getOpenFileName=lambda *args:(str(project),'')
        js("cmd('open');true");wait_for(w,f"state.dataset?.revision_id!=={json.dumps(prior)}&&!state.busy")
        assert c.business_config['measure']=='sales' and b.business.state()['result'] is None
        js("navigate('business');document.querySelector('[data-business-build]').click();true")
        wait_for(w,"!!state.business.result&&!state.busy")
    finally:QFileDialog.getSaveFileName,QFileDialog.getOpenFileName=old_save,old_open
    for theme in ['pastel','dark']:
        js(f"cmd('preferences',{{theme:'{theme}'}});true")
        wait_for(w,f"document.documentElement.dataset.theme==='{theme}'")
        for width,height in [(1440,900),(1024,768),(800,640)]:
            w.resize(width,height)
            for page in ['business','business_setup','business_segments','business_trends','business_pivot','business_scenario','business_data','business_report']:
                js(f"navigate('{page}');true")
                assert js("document.documentElement.scrollWidth<=innerWidth"),[theme,width,page]
                assert js("!!document.querySelector('.sidebar-dataset #remove-dataset')&&!document.querySelector('.topbar #remove-dataset')"),page
            js("navigate('business');true");loop=QEventLoop();QTimer.singleShot(250,loop.quit);loop.exec()
            w.grab().save(str(out/f'overview-{theme}-{width}.png'))
    js("document.querySelector('[data-workspace-mode=ml]').click();true")
    wait_for(w,"!state.business.enabled&&!!document.querySelector('#navigation [data-page=models]')")
    assert c.dataset.current.equals(f) and source.read_bytes()==original
    old_question=QMessageBox.question
    QMessageBox.question=lambda *args:QMessageBox.StandardButton.Yes
    try:
        js("document.querySelector('.sidebar-dataset #remove-dataset').click();true")
        wait_for(w,"state.dataset===null")
        assert source.read_bytes()==original and js("document.querySelector('.sidebar-dataset strong').textContent==='No dataset loaded'")
    finally:QMessageBox.question=old_question
    checks=['Dataset name and deletion moved below sidebar brand','Business mode toggles and restores ML navigation',
            'Real KPI/profit/segments/trends/pivot/scenario calculations','Segment click cross-filters and reset restores totals',
            'All six data/report exports and report definition open/save','Project reopen preserves report configuration and invalidates outputs',
            '48 layouts across 8 business routes, 3 sizes and 2 themes','Sidebar deletion preserves original source']
    result={'passed':True,'checks':checks};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    w.close();app.processEvents()


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
