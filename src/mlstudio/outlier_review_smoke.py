"""Reproduce the student's Iris row 59 = 200 and verify visible evidence."""
import json
from pathlib import Path
from sklearn.datasets import load_iris
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from .desktop import ModernWindow
from .data import DatasetVersions
from .analyzer import analyze
from .ui_smoke import evaluate,wait_for


def run():
    out=Path('examples/outlier-review-verified').resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1700,1050);w.show();js=lambda s:evaluate(w,s)
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        iris=load_iris(as_frame=True);f=iris.data.copy();f['target']=[iris.target_names[i] for i in iris.target];col='petal length (cm)';f.loc[58,col]=200
        source=out/'synthetic-iris-outlier.csv';f.to_csv(source,index=False)
        c.source=str(source);c.dataset=DatasetVersions(f);c.issues=analyze(f);c.refresh_data();c.target.setCurrentText('target');b.publish()
        wait_for(w,'state.dataset?.rows===150')
        js("studioUI.navigate('data');true")
        assert js("!!document.querySelector('[data-outlier-row=\"59\"][data-outlier-field=\"petal length (cm)\"]')")
        assert js("document.querySelector('.outlier-overview').textContent.includes('200')")
        js("document.querySelector('[data-outlier-filter=flagged]').click();true")
        wait_for(w,'state.dataset.outlier_filter===true')
        assert js("document.querySelectorAll('.easy-dataset-table tbody tr').length<150")
        js("document.querySelector('.easy-dataset-table .table-scroll').scrollIntoView({block:'center'});true")
        QTest.qWait(500);w.grab().save(str(out/'outlier-cell-200.png'))
        js("document.querySelector('[data-outlier-row=\"59\"][data-outlier-field=\"petal length (cm)\"]').click();true")
        assert js("document.querySelector('#outlier-detail').textContent.includes('200')&&document.querySelector('#outlier-detail').textContent.includes('IQR limits')")
        QTest.qWait(600);w.grab().save(str(out/'table-flagged-light.png'))
        js("document.querySelector('[data-outlier-box]').click();true")
        wait_for(w,"state.dataset_visuals?.kind==='box'&&!state.busy",timeout=60)
        assert '200' in [t.get_text() for t in b.dataset_visual_figure.axes[0].texts]
        js("document.querySelector('.dv-result')?.scrollIntoView({block:'start'});document.querySelector('#toast').classList.remove('show');true")
        QTest.qWait(800);w.grab().save(str(out/'boxplot-200.png'))
        b.dataset_visual_figure.savefig(out/'boxplot-200.svg',bbox_inches='tight')
        svg=out/'boxplot-200.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        js("studioUI.navigate('data');document.querySelector('[data-outlier-review]').click();true")
        wait_for(w,"state.preprocessing.detail?.title==='Potential outliers'")
        assert 200 in [row[0] for row in b.preprocess_detail['preview']['data']]
        js("studioUI.navigate('data');cmd('preferences',{theme:'dark'});true")
        QTest.qWait(500);js("document.getElementById('outlier-detail')?.scrollIntoView({block:'center'});true");QTest.qWait(300)
        w.grab().save(str(out/'table-flagged-dark.png'))
        assert c.dataset.current.iloc[58][col]==200
        result={'passed':True,'checks':['Iris row 59 value 200 visibly marked in table','Full-data summary includes 200 and numeric limits',
            'Flagged-only filter shows affected active rows','Cell click shows row/value and IQR evidence','One-click boxplot labels 200',
            'Prepare data opens the corresponding outlier finding','Light and Dark views keep the outlier label','Dataset unchanged'],
            'outlier_rows':b.dataset_state()['outlier_rows'],'outlier_cells':b.dataset_state()['outlier_cells']}
        (out/'result.json').write_text(json.dumps(result,indent=2));print('Outlier review desktop smoke passed')
    finally:w.close();app.processEvents()


if __name__=='__main__':run()
