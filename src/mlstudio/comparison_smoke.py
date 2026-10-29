"""Embedded acceptance for the competitor-gap tools with labeled synthetic data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication,QFileDialog
from .desktop import ModernWindow
from .ui_smoke import evaluate,wait_for,capture
from .arff_io import export_arff,load_arff
from .workflow import Graph
from .data import DatasetVersions


def run(directory):
    out=Path(directory).resolve();out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);w=ModernWindow(out/'storage');c,b=w.controller,w.bridge
    c.settings=QSettings(str(out/'settings.ini'),QSettings.Format.IniFormat);c.settings.setValue('referenceTheme','pastel')
    w.resize(1500,1000);w.show();js=lambda s:evaluate(w,s)
    notices=[];b.notice.connect(lambda s:notices.append(json.loads(s)))
    old_open,old_save=QFileDialog.getOpenFileName,QFileDialog.getSaveFileName
    try:
        wait_for(w,"document.body.dataset.ready==='true'")
        f=pd.DataFrame({'id':range(80),'measurement':np.arange(80,dtype=float),'group':['a','b']*40,
            'colour':['red','blue']*40,'label':[0,1]*40});f.loc[3,'measurement']=np.nan
        source=out/'synthetic.arff';export_arff(f,source)
        QFileDialog.getOpenFileName=lambda *a:(str(source),'')
        js("cmd('import');true");wait_for(w,"state.dataset?.rows===80&&!state.busy")
        assert list(c.dataset.current)==list(f)
        js("navigate('advanced');true")
        assert js("!!document.querySelector('[data-page=association]')&&!!document.querySelector('[data-page=model_diagnostics]')")
        js("navigate('association');true")
        js("document.querySelectorAll('[data-association-column]').forEach(n=>{n.checked=['group','colour'].includes(n.dataset.associationColumn);n.dispatchEvent(new Event('change',{bubbles:true}));});document.querySelector('#association-run').click();true")
        wait_for(w,"!!state.association?.result&&!state.busy")
        assert b.comparison_tools.rules['rules'] and js("document.querySelector('#content').textContent.includes('Lift')")
        rules=out/'rules.csv';QFileDialog.getSaveFileName=lambda *a:(str(rules),'')
        js("document.querySelector('[data-action=association_export]').click();true")
        assert len(pd.read_csv(rules))>0;capture(w,out,'association')
        js("cmd('simple_train',{target:'label',task:'classification',algorithm:'Logistic Regression'});true")
        wait_for(w,"!!state.model&&!state.busy",timeout=60)
        js("navigate('model_diagnostics');document.querySelector('#diagnostics-build').click();true")
        wait_for(w,"!!state.model_diagnostics&&!state.busy",timeout=60)
        assert len(b.comparison_tools.diagnostics['charts'])==3
        assert js("document.querySelectorAll('.diagnostic-charts img').length===3")
        table=out/'holdout-predictions.csv';QFileDialog.getSaveFileName=lambda *a:(str(table),'')
        js("document.querySelector('[data-diagnostics-table=predictions]').click();true")
        assert len(pd.read_csv(table))==20
        image=out/'roc.svg';QFileDialog.getSaveFileName=lambda *a:(str(image),'')
        js("document.querySelector('[data-diagnostics-image=roc]').click();true")
        assert '<svg' in image.read_text();capture(w,out,'model_diagnostics')
        (out/'model_diagnostics.png').replace(out/'classification-diagnostics.png')
        js("document.querySelector('#diagnostics-class').selectedIndex=0;document.querySelector('#diagnostics-build').click();true")
        wait_for(w,"state.model_diagnostics?.positive_class==='0.0'&&!state.busy")
        arff=out/'exported.arff';QFileDialog.getSaveFileName=lambda *a:(str(arff),'')
        js("navigate('reports');document.querySelector('[data-action=export_arff]').click();true")
        assert len(load_arff(arff))==80
        extra=out/'extra.csv';pd.DataFrame({'id':range(80),'score':range(100,180)}).to_csv(extra,index=False)
        nodes={'1':{'type':'Dataset','position':[40,100],'params':{'source':'file','path':str(source)}},
            '2':{'type':'Dataset','position':[40,260],'params':{'source':'file','path':str(extra)}},
            '3':{'type':'Merge data','position':[300,160],'params':{'left_key':'id','right_key':'id'}}}
        c.canvas.restore(Graph(nodes,[]));b.publish();js("navigate('workflow');true")
        wait_for(w,"state.flow.outputs['1']?.status==='ready'&&state.flow.outputs['2']?.status==='ready'&&!state.busy")
        js("flowConnect('1','3');true");wait_for(w,"state.flow.outputs['3']?.status==='waiting'&&!state.busy")
        js("flowConnect('2','3');true");wait_for(w,"state.flow.outputs['3']?.status==='ready'&&!state.busy")
        assert b.flow.engine.values['3'].frame.score.iloc[0]==100
        js("flowSelect('3');true");assert js("document.querySelector('.wf-settings').textContent.includes('Input 2')")
        capture(w,out,'workflow')
        js("flowAdd('Group by',500,160);true");wait_for(w,"!!state.workflow.nodes['4']&&!state.busy&&!state.flow.stale")
        js("flowConnect('3','4');true");wait_for(w,"!state.busy&&!state.flow.stale")
        js("flowSelect('4');document.querySelector('[data-flow-param=group]').value='group';document.querySelector('[data-flow-param=group]').dispatchEvent(new Event('change',{bubbles:true}));true")
        wait_for(w,"state.flow.outputs['4']?.status==='ready'&&!state.busy")
        assert b.flow.engine.values['4'].frame.iloc[:,1].tolist()==[40,40]
        js("flowAdd('Pivot table',700,160);true");wait_for(w,"!!state.workflow.nodes['5']&&!state.busy&&!state.flow.stale")
        js("flowConnect('3','5');true");wait_for(w,"!state.busy&&!state.flow.stale")
        js("flowSelect('5');flowChange(g=>Object.assign(g.nodes['5'].params,{group:'group',column:'colour'}));true")
        wait_for(w,"state.flow.outputs['5']?.status==='ready'&&!state.busy")
        assert len(b.flow.engine.values['5'].frame)==2
        js("flowAdd('Concatenate',900,160);true");wait_for(w,"!!state.workflow.nodes['6']&&!state.busy&&!state.flow.stale")
        js("flowConnect('1','6');true");wait_for(w,"!state.busy&&!state.flow.stale")
        js("flowConnect('2','6');true");wait_for(w,"state.flow.outputs['6']?.status==='ready'&&!state.busy")
        assert len(b.flow.engine.values['6'].frame)==160
        project=out/'combined-project.json';c.save_project(str(project))
        assert len(json.loads(project.read_text())['workflow']['edges'])==6
        js("cmd('simple_train',{target:'id',task:'regression',algorithm:'Ridge'});true")
        wait_for(w,"state.model?.metadata.algorithm==='Ridge'&&!state.busy",timeout=60)
        js("navigate('model_diagnostics');document.querySelector('#diagnostics-build').click();true")
        wait_for(w,"state.model_diagnostics?.task==='regression'&&!state.busy",timeout=60)
        assert len(b.comparison_tools.diagnostics['charts'])==2
        js("cmd('preferences',{theme:'dark'});true");wait_for(w,"document.documentElement.dataset.theme==='dark'")
        capture(w,out,'model_diagnostics');(out/'model_diagnostics.png').replace(out/'regression-diagnostics-dark.png')
        changed=c.dataset.current.copy();changed.loc[0,'id']=np.nan;c.dataset=DatasetVersions(changed);c.refresh_data();b.publish()
        assert b.comparison_tools.state()['association']['result'] is None
        assert b.comparison_tools.state()['model_diagnostics'] is None
        js("navigate('models');minimalModel.target='id';minimalModel.task='regression';render();true")
        assert js("!!document.querySelector('[data-review-answer-gaps]')")
        assert not [n for n in notices if n['kind']=='error' and 'Choose a grouping' not in n['message']],notices
        # Same displayed history index must not revive output from another branch.
        branch=pd.DataFrame({'x':range(80),'category':['A','A','B','B']*20,
            'colour':['red','blue']*40,'label':[0,1]*40})
        branch_source=out/'revision-source.csv';branch.to_csv(branch_source,index=False)
        original_bytes=branch_source.read_bytes()
        c.source=str(branch_source);c.dataset=DatasetVersions(branch);c.preparation=None
        c.canvas.restore(Graph());c.refresh_data();b.publish()
        js("cmd('preview',{operation:'Filter rows',column:'category',value:'A'});true")
        js("cmd('apply_preview');true");wait_for(w,"state.dataset?.version===1&&!state.busy")
        revision_a=c.dataset.revision_id
        js("navigate('association');cmd('association_run',{columns:['category','colour']});true")
        wait_for(w,"!!state.association?.result&&!state.busy")
        c.target.setCurrentText('label')
        c.canvas.restore(Graph({'1':{'type':'Dataset','position':[40,100],'params':{'source':'workspace'}},
            '2':{'type':'Data table','position':[240,100]},'3':{'type':'Decision Tree','position':[240,260]}},[['1','2'],['1','3']]))
        b.publish();js("navigate('workflow');true")
        wait_for(w,"state.flow.outputs['3']?.status==='ready'&&!state.busy&&!state.flow.stale")
        assert b.flow.engine.values['2'].frame.x.iloc[0]==0
        js("navigate('preprocessing');cmd('undo');true");wait_for(w,"state.dataset?.version===0&&!state.busy")
        js("cmd('preview',{operation:'Filter rows',column:'category',value:'B'});true")
        js("cmd('apply_preview');true");wait_for(w,"state.dataset?.version===1&&!state.busy")
        assert c.dataset.revision_id!=revision_a and b.comparison_tools.state()['association']['result'] is None
        js("navigate('workflow');true")
        wait_for(w,"state.flow.outputs['3']?.status==='ready'&&!state.busy&&!state.flow.stale")
        assert b.flow.engine.values['2'].frame.x.iloc[0]==2
        revised=out/'revision-output.csv';QFileDialog.getSaveFileName=lambda *a:(str(revised),'')
        js("cmd('flow_export',{id:'2'});true")
        assert pd.read_csv(revised).x.iloc[0]==2
        js("cmd('flow_activate',{id:'3'});true")
        assert c.result.metadata['dataset_fingerprint']==b.flow.engine.values['3'].metadata['dataset_fingerprint']
        current_revision=c.dataset.revision_id
        js("navigate('preprocessing');cmd('undo');true");wait_for(w,"state.dataset?.version===0&&!state.busy")
        js("cmd('redo');true");wait_for(w,"state.dataset?.version===1&&!state.busy")
        assert c.dataset.revision_id==current_revision and branch_source.read_bytes()==original_bytes
        result={'passed':True,'checks':['Dense ARFF import through real dialog','Association controls, exact rules and full CSV export',
            'Three held-out diagnostic charts and prediction/SVG export','ARFF export from Reports',
            'Two-input merge via canvas connections and input-order controls','Group-by, pivot and two-input concatenation outputs',
            'Classification class selection and regression diagnostics in dark mode','Combined graph project save',
            'Dataset changes invalidate rules and diagnostics','Missing outcome preflight links to review',
            'Replacement cleaning branch cannot revive association rules',
            'Workflow recomputes replacement revision, exports current rows and activates current model',
            'Undo/redo retains revision identity and preserves source bytes']}
        from .correctness_smoke import persistence,evidence_contracts
        result['checks'].extend(persistence(w,out))
        result['checks'].extend(evidence_contracts(w,out))
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        QFileDialog.getOpenFileName,QFileDialog.getSaveFileName=old_open,old_save
        if not c.busy:w.close()
        app.processEvents()


if __name__=='__main__':run('examples/comparison-tools-verified')
