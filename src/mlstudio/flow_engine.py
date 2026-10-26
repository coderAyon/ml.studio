"""Incremental typed flow execution. GUI positions never invalidate computed values."""
from dataclasses import dataclass
import base64
import hashlib
import io
import json
import copy
import numpy as np
import pandas as pd
from .workflow import Graph,PORTS
from .training import train,compare,TrainingResult,REGRESSION,prediction_inputs
from .training_prep import validate_preparation

CATALOG=[('Dataset','Import CSV','Data','database'),('Data table','Data table','Data','table'),
 ('Select columns','Select columns','Data','columns'),('Remove duplicates','Remove duplicates','Prepare','filter'),
 ('Impute missing','Missing values','Prepare','filter'),('Normalize','Normalize / scale','Prepare','layers'),
 ('Prepare training','Preprocess','Prepare','settings'),('Prepared output','Prepared output','Prepare','table'),
 ('Histogram','Histogram','Visualize','chart'),('Scatter plot','Scatter plot','Visualize','chart'),
 ('Box plot','Box plot','Visualize','chart'),('Bar chart','Bar chart','Visualize','chart'),('Correlation','Correlation','Visualize','chart'),
 ('Train model','Train & test','Learn','brain'),('Compare models','Compare models','Learn','layers'),('Results','Results','Learn','check')]
KINDS={row[0] for row in CATALOG}


def validate_flow(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('nodes'),dict) or not isinstance(payload.get('edges'),list):raise ValueError('Invalid workflow graph.')
    nodes,edges=payload['nodes'],payload['edges']
    if len(nodes)>60 or len(edges)>120:raise ValueError('Use at most 60 steps and 120 connections.')
    for key,node in nodes.items():
        if not isinstance(key,str) or not key.isdigit() or not isinstance(node,dict) or node.get('type') not in KINDS:raise ValueError('Unavailable workflow step.')
        pos=node.get('position')
        if not isinstance(pos,list) or len(pos)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=10000 for v in pos):raise ValueError('Keep nodes inside the canvas.')
        if not isinstance(node.get('params',{}),dict) or len(json.dumps(node.get('params',{})))>20000:raise ValueError('Invalid node settings.')
    if len({tuple(edge) for edge in edges if isinstance(edge,list) and len(edge)==2})!=len(edges):raise ValueError('Connections must be distinct pairs.')
    graph=Graph(copy.deepcopy(nodes),copy.deepcopy(edges))
    if nodes:graph.order(allow_disconnected=True)
    return graph


def table(frame,limit=40):
    return json.loads(frame.iloc[:limit,:30].to_json(orient='split',date_format='iso'))


@dataclass
class Packet:
    frame:pd.DataFrame
    preparation:dict


class FlowEngine:
    def __init__(self):self.cache={};self.values={};self.result={'nodes':{},'computed':0,'cached':0};self.fitted=[]

    def run(self,payload,frame,target,algorithm,preparation,dataset_key,progress=None,evaluation_error=None):
        graph=validate_flow(payload);order=graph.order(allow_disconnected=True) if graph.nodes else []
        if frame is not None and (len(frame)>20000 or len(frame.columns)>300):raise ValueError('Workflow supports up to 20,000 rows and 300 columns. Use a smaller extract.')
        defaults=validate_preparation(preparation);nodes={};values={};keys={};computed=cached=0;fitted=[]
        for key in order:
            node=graph.nodes[key];kind=node['type'];parents=[a for a,b in graph.edges if b==key];parent=parents[0] if parents else None
            if progress:progress(key)
            if parent and nodes[parent]['status']!='ready':nodes[key]={'status':'blocked','message':'Waiting for the connected step to succeed.'};continue
            if kind!='Dataset' and not parent:nodes[key]={'status':'waiting','message':'Connect an input to this step.'};continue
            if kind=='Dataset' and frame is None:nodes[key]={'status':'waiting','message':'Choose a CSV using this step’s Import button.'};continue
            signature=hashlib.sha256(json.dumps([kind,node.get('params',{}),keys.get(parent),str(dataset_key) if kind=='Dataset' else None,
                target if kind in {'Train model','Compare models','Prepared output'} else None,
                algorithm if kind in {'Train model','Compare models','Prepared output'} else None,defaults if kind in {'Dataset','Prepare training'} else None,
                evaluation_error if kind in {'Train model','Compare models','Prepared output'} else None],sort_keys=True).encode()).hexdigest()
            keys[key]=signature
            try:
                if signature in self.cache:
                    value,summary=self.cache[signature];summary=copy.deepcopy(summary);summary['cached']=True;cached+=1
                else:
                    incoming=values.get(parent)
                    if evaluation_error and kind in {'Train model','Compare models','Prepared output'}:raise ValueError(evaluation_error)
                    value,summary=self.compute(kind,incoming,frame,target,algorithm,defaults,node.get('params',{}))
                    summary={'status':'ready',**summary};self.cache[signature]=(value,copy.deepcopy(summary));computed+=1
                    if kind=='Train model':fitted.append(value)
                    elif kind=='Compare models':fitted.extend(value)
                values[key]=value;nodes[key]=summary
            except Exception as exc:nodes[key]={'status':'error','message':str(exc)}
        self.cache={key:value for key,value in self.cache.items() if key in set(keys.values())}
        self.values=values;self.fitted=fitted
        self.result={'nodes':nodes,'computed':computed,'cached':cached,'ready':sum(v['status']=='ready' for v in nodes.values()),'errors':sum(v['status']=='error' for v in nodes.values())}
        return self.result

    def compute(self,kind,incoming,frame,target,algorithm,defaults,params):
        if kind=='Dataset':value=Packet(frame.copy(deep=True),copy.deepcopy(defaults))
        elif isinstance(incoming,Packet):
            value=Packet(incoming.frame,copy.deepcopy(incoming.preparation));f=value.frame
            if kind=='Select columns':
                columns=params.get('columns',list(f.columns))
                if not isinstance(columns,list) or not columns or len(columns)!=len(set(columns)) or any(c not in f for c in columns):raise ValueError('Choose one or more distinct existing columns.')
                value.frame=f[columns].copy();value.preparation['excluded']=[c for c in value.preparation['excluded'] if c in columns]
            elif kind=='Remove duplicates':value.frame=f.drop_duplicates().reset_index(drop=True)
            elif kind in {'Impute missing','Normalize','Prepare training'}:
                if kind=='Impute missing':overrides={'imputation':params.get('imputation','median')}
                elif kind=='Normalize':overrides={'scaling':params.get('scaling','standard')}
                else:overrides=params.get('preparation',{})
                value.preparation=validate_preparation({**value.preparation,**overrides},f.columns)
            elif kind in {'Train model','Compare models','Prepared output'}:
                selected_target=params.get('target') or target;selected_algorithm=params.get('algorithm') or algorithm
                if kind=='Prepared output':return self.prepared(value,selected_target,selected_algorithm)
                if not selected_target or selected_target not in f:raise ValueError('Choose an outcome column in this step’s settings.')
                if kind=='Compare models':
                    candidates=params.get('algorithms') or (['Ridge','Decision Tree Regressor','Random Forest Regressor'] if selected_algorithm in REGRESSION else ['Logistic Regression','Decision Tree','Random Forest'])
                    if not isinstance(candidates,list) or not 1<=len(candidates)<=6:raise ValueError('Choose 1–6 comparison methods.')
                    result=compare(f,selected_target,candidates,preparation=value.preparation)
                    return result,{'kind':'comparison','runs':[{'metrics':r.metrics,'metadata':r.metadata} for r in result]}
                result=train(f,selected_target,selected_algorithm,preparation=value.preparation,parameters=params.get('parameters'))
                if kind=='Train model':return result,{'kind':'model','model':{'metrics':result.metrics,'metadata':result.metadata}}
            elif kind in {'Histogram','Scatter plot','Box plot','Bar chart','Correlation'}:
                from .dataset_visuals import build_chart
                numeric=list(f.select_dtypes(include='number'));chartkind={'Histogram':'histogram','Scatter plot':'scatter','Box plot':'box','Bar chart':'bar','Correlation':'correlation'}[kind]
                count=2 if chartkind in {'scatter','correlation'} else 1
                columns=params.get('columns') or numeric[:count]
                config={'kind':chartkind,'columns':columns,'group':params.get('group') or next((c for c in f if c not in numeric),f.columns[0]),'statistic':'count','bins':20,'outlier_method':'iqr'}
                if chartkind!='bar' and len(columns)<count:raise ValueError('Choose enough numeric measurements in the step settings.')
                result,figure=build_chart(f,config);buffer=io.BytesIO();figure.savefig(buffer,format='svg',bbox_inches='tight')
                return result,{'kind':'chart','chart':'data:image/svg+xml;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                    'title':result['title'],'table':result['table'],'note':result['protocol']}
            elif kind!='Data table':raise ValueError('Unsupported data step.')
        elif kind=='Results' and (isinstance(incoming,TrainingResult) or isinstance(incoming,list)):
            if isinstance(incoming,TrainingResult):return incoming,{'kind':'model','model':{'metrics':incoming.metrics,'metadata':incoming.metadata}}
            return incoming,{'kind':'comparison','runs':[{'metrics':r.metrics,'metadata':r.metadata} for r in incoming]}
        else:raise ValueError('This connection does not supply the expected output.')
        return value,{'kind':'data','rows':len(value.frame),'columns':len(value.frame.columns),'fields':list(value.frame.columns),
            'numeric':list(value.frame.select_dtypes(include='number').columns),'table':table(value.frame),'preparation':value.preparation,
            'note':'Original values are shown. Learned missing-value filling and scaling are deferred until the training split. Add Prepared output to inspect the transformed training values.'}

    def prepared(self,packet,target,algorithm):
        from sklearn.model_selection import train_test_split
        from .training import build_pipeline
        f=packet.frame;p=validate_preparation(packet.preparation,f.columns,target or None)
        if len(f)<4:raise ValueError('Prepared output needs at least four rows to separate training and preview holdout records.')
        if target and target not in f:raise ValueError('Choose an existing outcome or leave it unselected.')
        if p['feature_selection']=='f_test' and not target:raise ValueError('F-test selection needs an outcome. Choose it in this step, or turn off supervised selection.')
        columns=[c for c in f if c!=target and c not in p['excluded']]
        if not columns:raise ValueError('Keep at least one input column.')
        x=f[columns].copy();numeric=list(x.select_dtypes(include='number'))
        if numeric and np.isinf(x[numeric].to_numpy(dtype=float,na_value=np.nan)).any():raise ValueError('Replace infinite values with missing before preparing these inputs.')
        width=len(numeric)+sum(min(100,x[c].nunique(dropna=False)) for c in x if c not in numeric)
        if len(f)*max(1,width)>5_000_000:raise ValueError('Prepared inputs exceed the memory budget. Exclude unused/high-cardinality columns.')
        for c in x:
            if c not in numeric:x[c]=x[c].map(lambda v:str(v) if pd.notna(v) else np.nan).astype(object)
        y=f[target] if target else None
        if p['feature_selection']=='f_test' and y.isna().any():raise ValueError('Review missing outcome labels before supervised feature selection.')
        stratify=y if target and algorithm not in REGRESSION and not y.isna().any() and y.nunique()>1 and y.value_counts().min()>=2 and y.nunique()<=int(np.ceil(len(f)*.25)) else None
        train_positions,test_positions=train_test_split(np.arange(len(f)),test_size=.25,random_state=42,stratify=stratify)
        pipeline=build_pipeline(x.iloc[train_positions],'Baseline Regressor' if algorithm in REGRESSION else 'Baseline classifier',p)[:-1]
        pipeline.fit(x.iloc[train_positions],y.iloc[train_positions] if y is not None else None)
        names=pipeline.get_feature_names_out().tolist();transformed=pipeline.transform(x.iloc[train_positions[:30]])
        prepared=pd.DataFrame(transformed,columns=names)
        return prepared,{'kind':'prepared','table':table(prepared),'training_rows':len(train_positions),'test_rows':len(test_positions),
            'output_columns':len(names),'preparation':p,'note':('No outcome selected; all chosen columns are treated as inputs. ' if not target else 'The selected outcome is excluded from inputs. ')+
            'Preview of prepared TRAINING rows only (up to 30 rows / 30 inputs shown). Statistics were fitted without holdout rows. No predictor is trained by this preview; this terminal view never sends fitted full-data statistics into evaluation.'}
