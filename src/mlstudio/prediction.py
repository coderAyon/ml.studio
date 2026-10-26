"""Reviewable inference on explicit input mappings with optional probability columns."""
import numpy as np
import pandas as pd
from .training import prediction_inputs


def prediction_table(result,source,mapping=None,allow_extra=False):
    expected=result.metadata['features']
    if not 1<=len(source)<=20000:raise ValueError('Prediction preview supports 1–20,000 records. Use a smaller CSV extract.')
    mapping={c:c for c in expected} if mapping is None else mapping
    if not isinstance(mapping,dict) or set(mapping)!=set(expected) or any(not isinstance(v,str) or v not in source for v in mapping.values()) or len(set(mapping.values()))!=len(mapping):
        raise ValueError('Map each required input to a different existing CSV column.')
    unused=[c for c in source if c not in mapping.values()]
    if unused and not allow_extra:raise ValueError('CSV contains extra columns. Explicitly allow keeping them without using them as model inputs.')
    frame=pd.DataFrame({name:source[mapping[name]].to_numpy() for name in expected})
    x=prediction_inputs(result,frame);pred=result.pipeline.predict(x);out=source.copy(deep=True)
    def new_name(name):
        while name in out:name='_'+name
        return name
    label=new_name('prediction');out[label]=pred
    columns=[]
    if hasattr(result.pipeline,'predict_proba'):
        if len(source)*len(result.pipeline.named_steps['model'].classes_)>2_000_000:raise ValueError('Probability output exceeds the local memory budget. Use a smaller prediction CSV.')
        probability=result.pipeline.predict_proba(x)
        for j,cls in enumerate(result.pipeline.named_steps['model'].classes_):
            name=new_name('probability_'+str(cls));out[name]=probability[:,j];columns.append(name)
    return out,{'rows':len(out),'prediction_column':label,'probabilities':bool(columns),'probability_columns':columns,
        'unused_columns':unused,'mapping':mapping,'note':'Model probabilities are estimates, not guarantees or calibrated confidence. Models without predict_proba return labels/values only. No training is performed on this CSV.'}
