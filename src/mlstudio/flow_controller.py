"""GUI coordination for incremental auto-running canvas workflows."""
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QFileDialog
from .flow_engine import FlowEngine,Packet,CATALOG,validate_flow
from .training import TrainingResult
from .training_prep import assert_evaluation_ready


class FlowCoordinator(QObject):
    progress=Signal(str)

    def __init__(self,bridge):
        super().__init__(bridge);self.bridge=bridge;self.engine=FlowEngine();self.public={'nodes':{},'ready':0,'errors':0,'computed':0,'cached':0}
        self.last_signature=None;self.auto=True;self.progress.connect(self.on_progress)
        self.source_cache={}

    def signature(self):
        c=self.bridge.controller;g=c.canvas.snapshot()
        nodes={k:{'type':n['type'],'params':n.get('params',{})} for k,n in g.nodes.items()}
        data=[id(c.dataset),c.dataset.index if c.dataset else None,nodes,g.edges,c.target.currentText(),c.algorithm.currentText(),self.bridge.training_preparation()]
        for node in g.nodes.values():
            p=node.get('params',{})
            if node['type']=='Dataset' and p.get('source')=='file':
                try:stat=Path(p.get('path','')).stat();data.append([p.get('path'),stat.st_size,stat.st_mtime_ns])
                except (OSError,TypeError,ValueError):data.append([p.get('path'),'unavailable'])
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    def state(self):
        stale=self.signature()!=self.last_signature
        return {'catalog':[{'type':k,'label':l,'group':g,'icon':ic} for k,l,g,ic in CATALOG],
            'outputs':{} if stale else self.public['nodes'],'last_outputs':self.public['nodes'] if not stale else {},
            'ready':self.public.get('ready',0) if not stale else 0,'errors':self.public.get('errors',0) if not stale else 0,
            'computed':self.public.get('computed',0),'cached':self.public.get('cached',0),'stale':stale,'auto':self.auto,
            'ports':{k:list(__import__('mlstudio.workflow',fromlist=['PORTS']).PORTS[k]) for k,_,_,_ in CATALOG}}

    def on_progress(self,key):
        # A lightweight progress message avoids replacing the DOM while a native drag is active.
        self.bridge.on_status('Workflow · '+key+' · computing connected step…')

    def command(self,action,data):
        c=self.bridge.controller
        if action in {'flow_import','flow_use_workspace'}:
            key=str(data.get('id'));node=c.canvas.graph.nodes.get(key)
            if not node or node['type']!='Dataset':raise ValueError('Select an Import CSV node first.')
            if action=='flow_use_workspace':
                c.require_data();node['params']={'source':'workspace'};self.run();return
            path,_=QFileDialog.getOpenFileName(self.bridge.parent(),'Choose CSV for this node','','CSV and text (*.csv *.tsv *.txt)')
            if not path:return
            from .data import load_csv,DatasetVersions
            from .analyzer import analyze
            def read():
                frame=load_csv(path)
                if len(frame)>20000 or len(frame.columns)>300:raise ValueError('A workflow CSV supports up to 20,000 rows and 300 columns.')
                return frame
            def loaded(frame):
                node['params']={'source':'file','path':str(Path(path).resolve())}
                stat=Path(path).stat();self.source_cache[str(Path(path).resolve())]=(stat.st_size,stat.st_mtime_ns,frame)
                if c.dataset is None:
                    c.source=path;c.import_options={};c.dataset=DatasetVersions(frame);c.issues=analyze(frame);c.preparation=None;c.refresh_data()
                self.last_signature=None
            self.bridge.research._background(read,'CSV selected for this node. Connected steps will update.',loaded);return
        if action=='flow_commit':
            graph=validate_flow(data.get('graph'));c.canvas.restore(graph)
            self.auto=bool(data.get('auto',self.auto))
            if self.auto:self.run()
        elif action=='flow_run':self.run(force=bool(data.get('force',False)))
        elif action=='flow_impute':
            if self.signature()!=self.last_signature:raise ValueError('Wait for the connected steps to update before clicking Impute.')
            key=str(data.get('id'));graph=c.canvas.snapshot()
            payload={'nodes':graph.nodes,'edges':graph.edges}
            approval=self.engine.imputation_approval(payload,key,c.target.currentText(),c.algorithm.currentText())
            c.canvas.graph.nodes[key].setdefault('params',{})['imputation_approval']=approval
            self.run()
        elif action=='flow_auto':self.auto=bool(data.get('enabled',True))
        elif action=='flow_activate':
            if self.signature()!=self.last_signature:raise ValueError('Workflow changed. Wait for the connected steps to finish.')
            value=self.engine.values.get(str(data.get('id')))
            if not isinstance(value,TrainingResult):raise ValueError('Choose a completed model or Results step.')
            c.result=value;self.bridge.announce('Workflow model is active. Open prediction tools to use or save it.')
        elif action=='flow_export':
            if self.signature()!=self.last_signature:raise ValueError('Run the current workflow before exporting its output.')
            value=self.engine.values.get(str(data.get('id')))
            if isinstance(value,Packet):frame=value.imputed_preview if value.imputed_preview is not None else value.frame
            elif isinstance(value,pd.DataFrame):frame=value
            elif isinstance(value,dict) and 'table' in value:frame=pd.DataFrame(value['table']['data'],columns=value['table']['columns'])
            elif isinstance(value,list):frame=pd.DataFrame([{'algorithm':r.metadata['algorithm'],**{k:v for k,v in r.metrics.items() if isinstance(v,(int,float,str))}} for r in value])
            elif isinstance(value,TrainingResult):frame=pd.DataFrame([{k:v for k,v in value.metrics.items() if isinstance(v,(int,float,str))}])
            else:raise ValueError('This step has no output yet.')
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Export step output','workflow-output.csv','CSV (*.csv)')
            if path:c.protect_output(path);frame.to_csv(path,index=False)
        else:raise ValueError('Unknown workflow action.')

    def run(self,force=False):
        c=self.bridge.controller;signature=self.signature()
        graph=copy.deepcopy(c.canvas.snapshot());payload={'nodes':graph.nodes,'edges':graph.edges}
        if not graph.nodes:self.public={'nodes':{},'ready':0,'errors':0};self.engine=FlowEngine();self.last_signature=signature;return
        if signature==self.last_signature and not force:return
        frame=c.dataset.current if c.dataset else None
        evaluation_error=None
        if c.dataset:
            try:assert_evaluation_ready(c.dataset)
            except ValueError as exc:evaluation_error=str(exc)
        target,algorithm=c.target.currentText(),c.algorithm.currentText();prep=self.bridge.training_preparation()
        dataset_key=(id(c.dataset),c.dataset.index if c.dataset else None)
        def work():
            try:
                from .data import load_csv
                sources={};versions={};active_paths=set()
                for key,node in graph.nodes.items():
                    p=node.get('params',{})
                    if node['type']=='Dataset' and p.get('source')=='file':
                        path=p.get('path','');active_paths.add(path)
                        try:
                            stat=Path(path).stat();version=(stat.st_size,stat.st_mtime_ns);cached=self.source_cache.get(path)
                            data=cached[2] if cached and cached[:2]==version and not force else load_csv(path)
                            if len(data)>20000 or len(data.columns)>300:raise ValueError('Workflow CSV exceeds 20,000 rows / 300 columns.')
                            self.source_cache[path]=(*version,data);sources[key]=data;versions[key]=[path,*version,id(data)]
                        except Exception as exc:sources[key]='CSV unavailable: '+str(exc);versions[key]=[path,'unavailable']
                self.source_cache={p:v for p,v in self.source_cache.items() if p in active_paths}
                if force:self.engine.cache.clear()
                return self.engine.run(payload,frame,target,algorithm,prep,{'workspace':dataset_key,'files':versions},self.progress.emit,evaluation_error,sources,True)
            except Exception as exc:
                self.engine.values={};self.engine.fitted=[]
                return {'nodes':{key:{'status':'error','message':str(exc)} for key in graph.nodes},'ready':0,'errors':len(graph.nodes),'computed':0,'cached':0}
        def complete(result):
            self.public=result;self.last_signature=signature
            for fitted in self.engine.fitted:
                c.store.add(fitted.metadata['algorithm'],{'metrics':fitted.metrics,'metadata':fitted.metadata,'origin':'Reactive workflow'})
            if self.engine.fitted:c.refresh_history()
        self.bridge.research._background(work,'Connected workflow steps updated. Click a step to inspect its output.',complete)
