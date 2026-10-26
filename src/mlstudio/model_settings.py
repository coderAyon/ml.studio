"""Bounded, human-editable estimator settings and a small training-only search."""
import numpy as np


def parameter_specs(algorithm):
    from .training import ALGORITHMS
    defaults=ALGORITHMS[algorithm]().get_params();spec={}
    for key,low,high in [('n_neighbors',1,99),('max_depth',1,50),('n_estimators',10,200),('C',.001,1000),('alpha',.0001,100),('l1_ratio',0,1)]:
        if key in defaults:spec[key]={'default':defaults[key],'min':low,'max':high,'integer':key in {'n_neighbors','max_depth','n_estimators'}}
    return spec


def model_parameters(algorithm,values=None):
    values={} if values is None else values;spec=parameter_specs(algorithm)
    if not isinstance(values,dict) or set(values)-set(spec):raise ValueError('Unsupported model parameter. Review model settings.')
    for key,v in values.items():
        s=spec[key]
        if v is None and key=='max_depth':continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not s['min']<=v<=s['max'] or (s['integer'] and type(v) is not int):
            raise ValueError(f'{key} must be {"an integer" if s["integer"] else "a number"} from {s["min"]} to {s["max"]}.')
    return dict(values)


def search_grid(algorithm,min_rows):
    specs=parameter_specs(algorithm)
    for key,values in [('n_neighbors',[1,3,5]),('C',[.1,1.,10.]),('alpha',[.01,.1,1.]),('max_depth',[3,6,None]),('n_estimators',[30,60,100])]:
        if key in specs:
            candidates=[v for v in values if key!='n_neighbors' or v<=min_rows]
            if len(candidates)<2:raise ValueError('Too few training rows for neighbor search. Add rows or turn off tuning.')
            return {'model__'+key:candidates}
    raise ValueError('Small parameter search is unavailable for this method. Choose editable settings or another method.')
