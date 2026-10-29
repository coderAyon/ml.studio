"""Inspect real holdout outcomes; explanatory evidence is never a training input."""
import numpy as np
import pandas as pd


def scalar(value):
    if pd.isna(value):return None
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value if isinstance(value,(str,int,float,bool)) else str(value)


def holdout_details(frame,features,positions,actual,predicted,pipeline,baseline,regression):
    actual=np.asarray(actual);predicted=np.asarray(predicted)
    errors=np.abs(actual.astype(float)-predicted.astype(float)) if regression else actual!=predicted
    order=np.argsort(-errors.astype(float),kind='stable')[:200];records=[]
    for i in order:
        pos=int(positions[i]);row={'position':pos,'csv_row':pos+1,'actual':scalar(actual[i]),'predicted':scalar(predicted[i]),
            'features':{c:scalar(frame.iloc[pos][c]) for c in features[:20]}}
        row.update({'absolute_error':float(errors[i])} if regression else {'correct':not bool(errors[i])})
        records.append(row)
    result={'task':'regression' if regression else 'classification','total':len(actual),'shown':len(records),
        'records':records,'baseline':baseline,'error_count':None if regression else int(errors.sum()),'roc':None,
        'note':'Actual holdout rows. Up to 200 records, errors first, and 20 input columns shown. Row numbers refer to the active dataset version used for this run. Repeated choices after inspecting these outcomes require new independent test data.'}
    model=pipeline.named_steps['model']
    if not regression and len(model.classes_)==2:
        from .training import prediction_inputs,TrainingResult
        x=prediction_inputs(TrainingResult(pipeline,{}, {'features':features}),frame.iloc[positions][features])
        from sklearn.metrics import roc_curve,roc_auc_score
        score=pipeline.predict_proba(x)[:,1] if hasattr(pipeline,'predict_proba') else pipeline.decision_function(x) if hasattr(pipeline,'decision_function') else None
        if score is not None and np.isfinite(score).all():
            truth=actual==model.classes_[1]
            if len(np.unique(truth))==2:
                fpr,tpr,_=roc_curve(truth,score);keep=np.unique(np.linspace(0,len(fpr)-1,min(200,len(fpr))).astype(int))
                result['roc']={'positive_class':str(model.classes_[1]),'auc':float(roc_auc_score(truth,score)),
                    'points':[[float(fpr[i]),float(tpr[i])] for i in keep]}
    return result


def explain_inputs(result,frame):
    from .training import prediction_inputs,REGRESSION
    from sklearn.inspection import permutation_importance
    from .holdout import validate_holdout
    features,target,positions=validate_holdout(result,frame)
    if len(features)>100:raise ValueError('Input explanation supports at most 100 raw features. Exclude unused inputs before training.')
    chosen=np.random.default_rng(42).choice(positions,min(500,len(positions)),replace=False)
    x=prediction_inputs(result,frame.iloc[chosen][features]);y=frame.iloc[chosen][target]
    regression=result.metadata['algorithm'] in REGRESSION
    if regression:y=pd.to_numeric(y,errors='raise')
    values=permutation_importance(result.pipeline,x,y,n_repeats=3,random_state=42,n_jobs=1,
        scoring='neg_root_mean_squared_error' if regression else 'f1_macro')
    rows=[{'feature':name,'mean':float(values.importances_mean[i]),'std':float(values.importances_std[i])} for i,name in enumerate(features)]
    rows.sort(key=lambda r:r['mean'],reverse=True)
    return {'rows':rows,'sample_rows':len(chosen),'repeats':3,'metric':'negative RMSE' if regression else 'Macro F1',
        'note':'Held-out permutation importance: drop in score after shuffling one original input. Positive means this model relied on that input; zero/negative is possible. Correlated inputs can hide each other. This is not causal importance; do not tune against repeatedly inspected holdout evidence.'}
