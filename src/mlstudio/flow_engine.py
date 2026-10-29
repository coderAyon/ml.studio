"""Incremental typed flow execution. GUI positions never invalidate computed values."""
from dataclasses import dataclass
import base64
import hashlib
import io
import json
import copy
import numpy as np
import pandas as pd
from .workflow import Graph,PORTS,MULTI_INPUTS
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
LEARNERS={'Decision Tree','Random Forest','SVM','KNN','Naive Bayes','Linear Regression','Ridge','Random Forest Regressor'}
EXTRA=[('Filter rows','Filter rows','Data','filter'),('Sample rows','Sample rows','Data','table'),('Drop missing rows','Drop missing rows','Prepare','filter'),
 ('Rename column','Rename column','Data','columns'),('Convert numeric','Convert to numeric','Prepare','settings'),('Trim whitespace','Trim whitespace','Prepare','filter'),
 ('Statistics','Column statistics','Explore','table'),('Outlier review','Outlier evidence','Explore','search'),('PCA','PCA inputs','Prepare','layers'),
 ('Select features','Rank/select inputs','Prepare','filter'),('Discretize','Bin measurements','Prepare','layers'),
 ('K-Means','K-Means','Cluster','layers'),('DBSCAN','DBSCAN','Cluster','layers'),('Hierarchical','Hierarchical','Cluster','layers')]
CATALOG.extend(EXTRA);CATALOG.extend((name,name,'Learn','brain') for name in sorted(LEARNERS));KINDS.update(row[0] for row in CATALOG)
CATALOG.extend((name,name,'Data','layers') for name in ['Merge data','Concatenate','Group by','Pivot table']);KINDS.update(row[0] for row in CATALOG)


def validate_flow(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('nodes'),dict) or not isinstance(payload.get('edges'),list):raise ValueError('Invalid workflow graph.')
    nodes,edges=payload['nodes'],payload['edges']
    if len(nodes)>60 or len(edges)>120:raise ValueError('Use at most 60 steps and 120 connections.')
    for key,node in nodes.items():
        if not isinstance(key,str) or not key.isdigit() or not isinstance(node,dict) or node.get('type') not in KINDS:raise ValueError('Unavailable workflow step.')
        pos=node.get('position')
        if not isinstance(pos,list) or len(pos)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=10000 for v in pos):raise ValueError('Keep nodes inside the canvas.')
        if not isinstance(node.get('params',{}),dict) or len(json.dumps(node.get('params',{})))>20000:raise ValueError('Invalid node settings.')
        if node['type']=='Dataset' and node.get('params',{}).get('source')=='file':
            path=node['params'].get('path')
            if not isinstance(path,str) or not path or len(path)>4096:raise ValueError('Choose a valid CSV path for this node.')
    if len({tuple(edge) for edge in edges if isinstance(edge,list) and len(edge)==2})!=len(edges):raise ValueError('Connections must be distinct pairs.')
    graph=Graph(copy.deepcopy(nodes),copy.deepcopy(edges))
    if nodes:graph.order(allow_disconnected=True)
    return graph


def table(frame,limit=40):
    return json.loads(frame.iloc[:limit,:30].to_json(orient='split',date_format='iso'))


def _clustering_columns(frame,target,params):
    return params.get('columns') or [c for c in frame.select_dtypes(include='number') if c!=target and not pd.api.types.is_bool_dtype(frame[c])]


@dataclass
class Packet:
    frame:pd.DataFrame
    preparation:dict
    evaluation_error:str|None=None
    imputed_preview:pd.DataFrame|None=None


class FlowEngine:
    def __init__(self):self.cache={};self.values={};self.result={'nodes':{},'computed':0,'cached':0};self.fitted=[];self.node_keys={}

    def imputation_approval(self,payload,key,target,algorithm):
        """Bind an explicit Impute click to the current input and method."""
        graph=validate_flow(payload);node=graph.nodes.get(key)
        if not node or node['type']!='Impute missing':raise ValueError('Choose a Missing values step.')
        parent=next((a for a,b in graph.edges if b==key),None)
        if not parent or self.result['nodes'].get(parent,{}).get('status')!='ready' or parent not in self.node_keys:
            raise ValueError('Connect a completed data step before clicking Impute.')
        return {'input':self.node_keys[parent],'method':node.get('params',{}).get('imputation','median'),
                'target':target,'algorithm':algorithm}

    def run(self,payload,frame,target,algorithm,preparation,dataset_key,progress=None,evaluation_error=None,sources=None,strict_sources=False):
        graph=validate_flow(payload);order=graph.order(allow_disconnected=True) if graph.nodes else []
        if not strict_sources and frame is not None and (len(frame)>20000 or len(frame.columns)>300):raise ValueError('Workflow supports up to 20,000 rows and 300 columns. Use a smaller extract.')
        defaults=validate_preparation(preparation);nodes={};values={};keys={};computed=cached=0;fitted=[]
        for key in order:
            node=graph.nodes[key];kind=node['type'];parents=[a for a,b in graph.edges if b==key];parent=parents[0] if parents else None
            source_frame=frame;source_error=evaluation_error
            if kind=='Dataset' and strict_sources:
                choice=node.get('params',{}).get('source')
                if choice=='file':
                    source_frame=(sources or {}).get(key);source_error=None
                    if isinstance(source_frame,str):nodes[key]={'status':'error','message':source_frame};continue
                elif choice!='workspace':nodes[key]={'status':'waiting','message':'Click Choose CSV, or explicitly use the current workspace dataset.'};continue
            if progress:progress(key)
            blocked=next((a for a in parents if nodes[a]['status']!='ready'),None)
            if blocked:
                message='Click Impute in the connected Missing values step to continue.' if nodes[blocked]['status']=='pending' else 'Waiting for the connected step to succeed.'
                nodes[key]={'status':'blocked','message':message};continue
            if kind in MULTI_INPUTS and len(parents)!=2:
                nodes[key]={'status':'waiting','message':'Connect two data outputs. The first connection is Input 1; the second is Input 2.'};continue
            if kind!='Dataset' and not parent:nodes[key]={'status':'waiting','message':'Connect an input to this step.'};continue
            if kind=='Dataset' and source_frame is None:nodes[key]={'status':'waiting','message':'Choose a CSV using this step’s Import button.'};continue
            if kind=='Dataset' and (len(source_frame)>20000 or len(source_frame.columns)>300):nodes[key]={'status':'error','message':'This source exceeds 20,000 rows / 300 columns. Choose a smaller CSV.'};continue
            root_key=dataset_key
            if kind=='Dataset' and isinstance(dataset_key,dict) and 'workspace' in dataset_key:
                root_key=dataset_key.get('files',{}).get(key) if node.get('params',{}).get('source')=='file' else dataset_key['workspace']
            signature_defaults=validate_preparation(None) if kind=='Dataset' and node.get('params',{}).get('source')=='file' else defaults
            signature_context=target if kind in {'Train model','Compare models','Prepared output','Normalize','Impute missing','Prepare training','PCA','Select features','Discretize'}|LEARNERS else None
            if kind in {'K-Means','DBSCAN','Hierarchical'} and isinstance(values.get(parent),Packet):
                signature_context=_clustering_columns(values[parent].frame,target,node.get('params',{}))
            signature=hashlib.sha256(json.dumps([kind,node.get('params',{}),[keys.get(a) for a in parents],str(root_key) if kind=='Dataset' else None,
                signature_context,
                algorithm if kind in {'Train model','Compare models','Prepared output','Normalize','Impute missing','Prepare training','PCA','Select features','Discretize'}|LEARNERS else None,signature_defaults if kind in {'Dataset','Prepare training'} else None,
                evaluation_error if kind in {'Train model','Compare models','Prepared output'} else None],sort_keys=True).encode()).hexdigest()
            keys[key]=signature
            if kind=='Impute missing':
                approval={'input':keys[parent],'method':node.get('params',{}).get('imputation','median'),
                          'target':target,'algorithm':algorithm}
                if node.get('params',{}).get('imputation_approval')!=approval:
                    incoming=values[parent]
                    missing_positions=np.flatnonzero(incoming.frame.isna().any(axis=1).to_numpy())[:30].tolist()
                    preview_positions=missing_positions+[i for i in range(len(incoming.frame)) if i not in missing_positions][:40-len(missing_positions)]
                    nodes[key]={'status':'pending','kind':'imputation','rows':len(incoming.frame),
                        'columns':len(incoming.frame.columns),'fields':list(incoming.frame.columns),
                        'numeric':list(incoming.frame.select_dtypes(include='number').columns),
                        'missing_cells':int(incoming.frame.isna().sum().sum()),'table':table(incoming.frame.iloc[preview_positions]),
                        'message':'Choose a filling method, then click Impute. No filling has been applied to this step.'}
                    continue
            try:
                if signature in self.cache:
                    value,summary=self.cache[signature];summary=copy.deepcopy(summary);summary['cached']=True;cached+=1
                else:
                    incoming=[values[a] for a in parents] if kind in MULTI_INPUTS else values.get(parent)
                    self.source_error=source_error
                    node_defaults=validate_preparation(None) if kind=='Dataset' and node.get('params',{}).get('source')=='file' else defaults
                    value,summary=self.compute(kind,incoming,source_frame,target,algorithm,node_defaults,node.get('params',{}))
                    summary={'status':'ready',**summary};self.cache[signature]=(value,copy.deepcopy(summary));computed+=1
                    if kind=='Train model' or kind in LEARNERS:fitted.append(value)
                    elif kind=='Compare models':fitted.extend(value)
                values[key]=value;nodes[key]=summary
            except Exception as exc:nodes[key]={'status':'error','message':str(exc)}
        self.cache={key:value for key,value in self.cache.items() if key in set(keys.values())}
        self.values=values;self.fitted=fitted;self.node_keys=keys
        self.result={'nodes':nodes,'computed':computed,'cached':cached,'ready':sum(v['status']=='ready' for v in nodes.values()),'errors':sum(v['status']=='error' for v in nodes.values())}
        return self.result

    def compute(self,kind,incoming,frame,target,algorithm,defaults,params):
        if kind in LEARNERS:params={**params,'algorithm':kind};kind='Train model'
        if kind in MULTI_INPUTS:
            from .table_operations import combine_tables
            if len(incoming)!=2 or any(not isinstance(v,Packet) for v in incoming):raise ValueError('Connect two completed tables.')
            joined=combine_tables(incoming[0].frame,incoming[1].frame,kind,params)
            errors=' '.join(v.evaluation_error for v in incoming if v.evaluation_error) or None
            packet=Packet(joined,validate_preparation(None),errors)
            return packet,{'kind':'data','rows':len(joined),'columns':len(joined.columns),'fields':list(joined.columns),
                'missing_cells':int(joined.isna().sum().sum()),'duplicate_rows':int(joined.duplicated().sum()),
                'numeric':list(joined.select_dtypes(include='number').columns),'table':table(joined),
                'note':'Input 1 and Input 2 follow connection order. Combined data uses a fresh preparation plan; review missing values, source columns and the outcome before learning.'}
        if kind=='Dataset':value=Packet(frame.copy(deep=True),copy.deepcopy(defaults),getattr(self,'source_error',None))
        elif isinstance(incoming,Packet):
            value=Packet(incoming.frame,copy.deepcopy(incoming.preparation),incoming.evaluation_error);f=value.frame
            if kind=='Select columns':
                columns=params.get('columns',list(f.columns))
                if not isinstance(columns,list) or not columns or len(columns)!=len(set(columns)) or any(c not in f for c in columns):raise ValueError('Choose one or more distinct existing columns.')
                value.frame=f[columns].copy();value.preparation['excluded']=[c for c in value.preparation['excluded'] if c in columns]
            elif kind=='Remove duplicates':value.frame=f.drop_duplicates().reset_index(drop=True)
            elif kind in {'Group by','Pivot table'}:
                from .table_operations import summarize_table
                value.frame=summarize_table(f,kind,params)
                value.evaluation_error='This branch summarizes all source rows. Train and evaluate from a branch before grouping or pivoting.'
                value.preparation=validate_preparation(None)
            elif kind in {'Filter rows','Drop missing rows','Rename column','Convert numeric','Trim whitespace'}:
                from .data import Transformation
                column=params.get('column') or f.columns[0]
                operation={'Convert numeric':'Convert to numeric'}.get(kind,kind)
                if kind=='Rename column' and column in value.preparation['excluded']:raise ValueError('Remove this column from saved exclusions before renaming it.')
                value.frame=Transformation(operation,[column],str(params.get('value',''))).preview(f)
                if value.frame.empty:raise ValueError('This step would leave no data. Review the column/value settings.')
            elif kind=='Sample rows':
                size=params.get('rows',min(100,len(f)))
                if type(size) is not int or not 1<=size<=len(f):raise ValueError('Choose a sample size between 1 and the input row count.')
                value.frame=f.sample(n=size,random_state=42).sort_index(kind='stable').reset_index(drop=True)
            elif kind in {'PCA','Select features','Discretize'}:
                overrides={'feature_selection':'pca','pca_components':params.get('components',2)} if kind=='PCA' else {'feature_selection':'f_test','selection_k':params.get('k',10)} if kind=='Select features' else {'discretization':params.get('strategy','quantile'),'bins':params.get('bins',5)}
                value.preparation=validate_preparation({**value.preparation,**overrides},f.columns)
            elif kind in {'Statistics','Outlier review'}:
                from .analyzer import profile
                from .preprocessing import iqr_evidence,_preview
                if kind=='Statistics':
                    fields=profile(f)['fields'];rows=[[p['name'],p['type'],p['missing'],p['unique'],p.get('mean'),p.get('min'),p.get('max'),p.get('outliers',{}).get('outlier_count')] for p in fields]
                    output={'columns':['Column','Type','Missing','Unique','Mean','Min','Max','IQR flags'],'data':rows,'index':list(range(len(rows)))}
                else:
                    columns=params.get('columns') or list(f.select_dtypes(include='number').columns)
                    columns=[c for c in columns if c in f and not pd.api.types.is_bool_dtype(f[c])]
                    if not columns:raise ValueError('Choose numeric measurements to check for outliers.')
                    mask=np.zeros(len(f),dtype=bool)
                    for c in columns:mask|=iqr_evidence(f,c)[0]
                    preview=_preview(f,np.flatnonzero(mask),columns);output={'columns':['Active row',*preview['columns']],'data':[[pos,*row] for pos,row in zip(preview['positions'],preview['data'])],'index':preview['index']}
                result={'table':output};return result,{'kind':'analysis','table':output,'rows':len(f),'note':'Statistics use all input rows. IQR flags are review findings, not confirmed mistakes. Evidence table is a bounded preview.'}
            elif kind in {'K-Means','DBSCAN','Hierarchical'}:
                from .clustering import study
                columns=_clustering_columns(f,target,params)
                result=study(f,columns,kind,clusters=params.get('clusters',3),eps=params.get('eps',.5),min_samples=params.get('min_samples',5))
                output=f.copy();name='Cluster'
                while name in output:name='_'+name
                output[name]=result['labels']
                clustered=Packet(output,value.preparation,'Clustering labels were learned from all input rows. Use this branch for exploration; train supervised models from a branch before clustering.')
                return clustered,{'kind':'analysis','table':table(output),'fields':list(output.columns),'numeric':list(output.select_dtypes(include='number').columns),'columns':len(output.columns),'metrics':result['metrics'],'rows':len(f),'note':result['protocol']}
            elif kind in {'Impute missing','Normalize','Prepare training'}:
                if kind=='Impute missing':overrides={'imputation':params.get('imputation','median')}
                elif kind=='Normalize':overrides={'scaling':params.get('scaling','standard')}
                else:overrides=params.get('preparation',{})
                value.preparation=validate_preparation({**value.preparation,**overrides},f.columns)
            elif kind in {'Train model','Compare models','Prepared output'}:
                if value.evaluation_error:raise ValueError(value.evaluation_error)
                selected_target=params.get('target') or target;selected_algorithm=params.get('algorithm') or algorithm
                if kind=='Prepared output':return self.prepared(value,selected_target,selected_algorithm,params)
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
        summary={'kind':'data','rows':len(value.frame),'columns':len(value.frame.columns),'fields':list(value.frame.columns),
            'missing_cells':int(value.frame.isna().sum().sum()),'duplicate_rows':int(value.frame.duplicated().sum()),
            'numeric':list(value.frame.select_dtypes(include='number').columns),'table':table(value.frame),'preparation':value.preparation,
            'note':'Original values are shown. Learned missing-value filling and scaling are deferred until the training split. Add Prepared output to inspect the transformed training values.'}
        if kind in {'Normalize','Impute missing','Prepare training','PCA','Select features','Discretize'}:
            try:
                preview_packet=value
                if kind=='Impute missing':
                    # This preview fills gaps only; other branch preparation runs later.
                    preview_plan={**value.preparation,'scaling':'none','outliers':'keep',
                        'missing_indicators':False,'discretization':'none','remove_constant':False,'feature_selection':'none'}
                    preview_packet=Packet(value.frame,validate_preparation(preview_plan),value.evaluation_error)
                _,preview=self.prepared(preview_packet,target if target in value.frame else '',algorithm,params,
                    include_imputation=kind=='Impute missing')
                summary['comparison']=preview['comparison']
                if kind=='Impute missing':
                    value.imputed_preview=preview_packet.imputed_preview
                    summary['comparison']['label']='missing-value filling'
                    summary['imputation_preview']=preview['imputation_preview']
            except ValueError as exc:
                if kind=='Impute missing':raise
                summary['preview_error']=str(exc)
        return value,summary

    def prepared(self,packet,target,algorithm,params=None,include_imputation=False):
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
        # Preserve positional identity, including holdout rows and duplicate index labels.
        # This is a display-only transform; downstream evaluation still fits its own plan.
        from .preprocessing import iqr_evidence,_json_value
        params=params or {};selected=params.get('preview_row')
        if selected is not None and (type(selected) is not int or not 1<=selected<=len(f)):
            raise ValueError('Choose an active row number inside this dataset.')
        chosen=params.get('preview_column')
        if chosen and chosen not in numeric:raise ValueError('Choose a numeric input column for value comparison.')
        flags={c:iqr_evidence(f,c)[0] for c in numeric if not pd.api.types.is_bool_dtype(f[c])}
        flagged=np.zeros(len(f),dtype=bool)
        for mask in flags.values():flagged|=mask
        missing=x.isna().any(axis=1).to_numpy()
        priority=np.flatnonzero(missing).tolist()+[int(i) for i in np.flatnonzero(flagged) if not missing[i]]
        positions=[selected-1] if selected else priority[:30]
        positions+=([] if selected else [i for i in range(len(f)) if i not in positions][:30-len(positions)])
        comparisons=[];effects=[]
        if numeric:
            all_numeric=pipeline.named_steps['preprocess'].named_transformers_['numeric'].transform(x[numeric])
            numeric_values=all_numeric[positions]
            train_set=set(train_positions.tolist())
            display_columns=[chosen] if chosen else numeric[:20]
            for c in display_columns:
                before=f[c].to_numpy(dtype=float,na_value=np.nan);after=all_numeric[:,numeric.index(c)]
                finite_before=before[np.isfinite(before)];finite_after=after[np.isfinite(after)]
                effects.append({'column':c,'before_missing':int(np.isnan(before).sum()),'after_missing':int(np.isnan(after).sum()),
                    'changed':int((~np.isclose(before,after,equal_nan=True)).sum()),
                    'before_min':_json_value(finite_before.min()) if len(finite_before) else None,'after_min':_json_value(finite_after.min()) if len(finite_after) else None,
                    'before_max':_json_value(finite_before.max()) if len(finite_before) else None,'after_max':_json_value(finite_after.max()) if len(finite_after) else None,
                    'before_mean':_json_value(finite_before.mean()) if len(finite_before) else None,'after_mean':_json_value(finite_after.mean()) if len(finite_after) else None})
            for j,pos in enumerate(positions):
                for c in display_columns:
                    comparisons.append({'row':int(pos)+1,'index':str(f.index[pos]),'column':c,
                        'before':_json_value(f[c].iloc[pos]),'after':_json_value(numeric_values[j,numeric.index(c)]),
                        'original_outlier':bool(flags[c][pos]) if c in flags else False,
                        'split':'Training' if pos in train_set else 'Holdout'})
        comparison={'cells':comparisons,'effects':effects,'output_columns':len(names),'training_rows':len(train_positions),'test_rows':len(test_positions),'total_rows':len(f),'numeric_columns':numeric,'scaling':p['scaling'],
            'selected_row':selected,'selected_column':chosen or '',
            'note':'Same active row before → after numeric preparation. Missing rows and original IQR outliers appear first and stay marked for tracking. Scaling changes units; it does not remove unusual observations. Statistics fit on training rows only; holdout values may fall outside 0–1. Values shown are before encoding/feature selection/PCA. Up to 30 rows × 20 numeric columns; choose a row and column to inspect any value.'}
        summary={'kind':'prepared','table':table(prepared),'training_rows':len(train_positions),'test_rows':len(test_positions),
            'comparison':comparison,'output_columns':len(names),'preparation':p,'note':('No outcome selected; all chosen columns are treated as inputs. ' if not target else 'The selected outcome is excluded from inputs. ')+
            'Preview of prepared TRAINING rows only (up to 30 rows / 30 inputs shown). Statistics were fitted without holdout rows. No predictor is trained by this preview; this terminal view never sends fitted full-data statistics into evaluation.'}
        if include_imputation:
            imputed=f.copy(deep=True)
            preprocess=pipeline.named_steps['preprocess']
            for branch,cols in [('numeric',numeric),('categorical',[c for c in x if c not in numeric])]:
                if not cols:continue
                filled=preprocess.named_transformers_[branch].named_steps['imputer'].transform(x[cols])
                for index,column in enumerate(cols):
                    absent=f[column].isna().to_numpy()
                    series=f[column].astype(float if column in numeric else object).copy()
                    series.iloc[np.flatnonzero(absent)]=filled[absent,index]
                    imputed[column]=series
            packet.imputed_preview=imputed
            summary['imputation_preview']={'table':table(imputed.iloc[positions]),
                'before_missing':int(f[columns].isna().sum().sum()),
                'after_missing':int(imputed[columns].isna().sum().sum()),
                'untouched_missing':int(imputed.isna().sum().sum()-imputed[columns].isna().sum().sum()),
                'method':p['imputation'],'rows':len(f),'fields':columns,
                'note':'Filled input preview. Original CSV stays unchanged. Outcome and excluded columns stay unchanged; model training fits filling on its own training split.'}
        return prepared,summary
