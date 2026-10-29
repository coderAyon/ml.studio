"""Offline, explicit single-table BI. Descriptive results never modify source data."""
import numpy as np
import pandas as pd

AGGREGATES={'count','sum','mean','median','min','max','distinct'}
GRAINS={'day':'D','week':'W-SUN','month':'M','quarter':'Q','year':'Y'}


def number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def clean_numeric(series):
    return series.replace([np.inf,-np.inf],np.nan)


def aggregate(frame, column, method):
    if method=='count':return len(frame)
    if method=='distinct':return int(frame[column].nunique(dropna=True))
    values=clean_numeric(frame[column]).dropna()
    if values.empty:return None
    # Aggregate in floating point, avoiding fixed-width integer sum wraparound.
    # These are analytical measurements, not decimal-exact accounting entries.
    values=values.astype('float64')
    with np.errstate(over='ignore',invalid='ignore'):
        return number(getattr(values,method)())


def _label_series(series):
    # Keep missing groups separate even when the source contains this literal label.
    labels=series.astype('string');missing='(missing)'
    existing=set(labels.dropna())
    while missing in existing:missing='_'+missing
    return labels.fillna(missing)


def _percent(current, previous):
    return number((current-previous)/abs(previous)*100) if current is not None and previous not in {None,0} else None


def analyze_business(frame, config):
    if not isinstance(frame,pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError('Import a nonempty dataset with unique column names.')
    if len(frame)>100000 or len(frame.columns)>300:raise ValueError('Business analysis supports up to 100,000 rows and 300 columns.')
    if not isinstance(config,dict):raise ValueError('Choose business report settings.')
    allowed={'measure','aggregate','dimension','date','grain','pivot','cost','customer','target','scenario_percent','filters','top_n','date_from','date_to','title'}
    if set(config)-allowed:raise ValueError('Unsupported business report setting.')
    cfg={'measure':'','aggregate':'count','dimension':'','date':'','grain':'month','pivot':'','cost':'','customer':'',
         'target':None,'scenario_percent':0.,'filters':[],'top_n':12,'date_from':'','date_to':'','title':'Business overview',**config}
    for key in ['measure','dimension','date','pivot','cost','customer']:
        if not isinstance(cfg[key],str) or (cfg[key] and cfg[key] not in frame):raise ValueError(f'Choose an existing {key} column.')
    method,measure=cfg['aggregate'],cfg['measure']
    if not isinstance(method,str) or method not in AGGREGATES:raise ValueError('Choose a supported measure aggregation.')
    if method!='count' and not measure:raise ValueError('Choose a measurement column.')
    def numeric(column):
        return column and pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    if method not in {'count','distinct'} and not numeric(measure):raise ValueError('This aggregation needs a numeric measurement.')
    if cfg['cost'] and (not numeric(measure) or not numeric(cfg['cost']) or method!='sum'):
        raise ValueError('Profit requires explicitly selected numeric revenue and cost columns with Sum aggregation.')
    if cfg['pivot'] and (not cfg['dimension'] or cfg['pivot']==cfg['dimension']):raise ValueError('Pivot needs two different grouping columns.')
    if not isinstance(cfg['grain'],str) or cfg['grain'] not in GRAINS:raise ValueError('Choose day, week, month, quarter or year.')
    if type(cfg['top_n']) is not int or not 1<=cfg['top_n']<=30:raise ValueError('Show 1–30 segments.')
    for key in ['target','scenario_percent']:
        v=cfg[key]
        if (key=='target' and v is None):continue
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError(f'{key} must be a finite number.')
    if not -100<=cfg['scenario_percent']<=1000:raise ValueError('What-if change must be between -100% and 1000%.')
    if not isinstance(cfg['title'],str) or len(cfg['title'])>160:raise ValueError('Use a report title up to 160 characters.')
    if any(not isinstance(cfg[key],str) for key in ['date_from','date_to']):raise ValueError('Choose date range values as date text.')
    filters=cfg['filters']
    if not isinstance(filters,list) or len(filters)>8:raise ValueError('Use up to eight AND filters.')
    mask=np.ones(len(frame),dtype=bool)
    for item in filters:
        if not isinstance(item,dict) or set(item)-{'column','op','value'}:raise ValueError('Invalid business filter.')
        col,op,v=item.get('column'),item.get('op'),item.get('value')
        if not isinstance(col,str) or col not in frame:raise ValueError('Choose a filter column.')
        s=frame[col]
        if op in {'missing','present'}:m=s.isna() if op=='missing' else s.notna()
        elif op in {'eq','ne','contains'}:
            if not isinstance(v,(str,int,float,bool)) or (isinstance(v,float) and not np.isfinite(v)):raise ValueError('Use a finite scalar filter value.')
            if op=='contains':m=s.astype('string').str.contains(str(v),regex=False,case=False,na=False)
            elif numeric(col):
                try:v=float(v)
                except (TypeError,ValueError) as exc:raise ValueError('Numeric equality needs a number.') from exc
                if not np.isfinite(v):raise ValueError('Use a finite filter value.')
                m=s.eq(v) if op=='eq' else s.ne(v)&s.notna()
            else:m=s.astype('string').eq(str(v)) if op=='eq' else s.astype('string').ne(str(v))&s.notna()
        elif op in {'gt','ge','lt','le'}:
            if not numeric(col):raise ValueError('Numeric range filters need a numeric column.')
            try:v=float(v)
            except (TypeError,ValueError) as exc:raise ValueError('Enter a numeric filter value.') from exc
            if not np.isfinite(v):raise ValueError('Use a finite filter value.')
            m=getattr(s,op)(v)&np.isfinite(s)
        else:raise ValueError('Choose a supported filter operator.')
        mask &= m.fillna(False).to_numpy(dtype=bool)
    dates=None
    if cfg['date']:
        if pd.api.types.is_numeric_dtype(frame[cfg['date']]):raise ValueError('Choose explicit date/time text or a datetime column, not ambiguous numeric timestamps.')
        dates=pd.to_datetime(frame[cfg['date']],errors='coerce',utc=True,format='mixed')
        for key,op in [('date_from','ge'),('date_to','le')]:
            if cfg[key]:
                try:bound=pd.to_datetime(cfg[key],utc=True,errors='raise')
                except (ValueError,TypeError) as exc:raise ValueError('Choose a valid date range.') from exc
                if pd.isna(bound):raise ValueError('Choose a valid date range.')
                if key=='date_to':bound=bound+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
                mask &= getattr(dates,op)(bound).to_numpy()
    elif cfg['date_from'] or cfg['date_to']:raise ValueError('Select a date column before filtering dates.')
    positions=np.flatnonzero(mask);f=frame.iloc[positions].copy(deep=True)
    if f.empty:raise ValueError('No records match these filters. Clear or change the filters.')
    value=aggregate(f,measure,method)
    observed=f[measure].notna() if measure else pd.Series(True,index=f.index)
    finite=clean_numeric(f[measure]).notna() if numeric(measure) else observed
    metric={'column':measure,'aggregate':method,'value':value,'finite_rows':int(finite.sum()),
            'missing_rows':int(f[measure].isna().sum()) if measure else 0,
            'infinite_rows':int(f[measure].isin([np.inf,-np.inf]).sum()) if numeric(measure) else 0}
    segments=None; pivot=None; trend=None; finance=None
    if cfg['dimension']:
        labels=_label_series(f[cfg['dimension']]);groups=f.groupby(labels,observed=True,sort=False)
        if groups.ngroups>500:raise ValueError('Choose a segment field with at most 500 groups; identifiers usually are unsuitable.')
        rows=[[str(label),aggregate(part,measure,method),len(part)] for label,part in groups]
        rows.sort(key=lambda r:(r[1] is not None,r[1] if r[1] is not None else 0),reverse=True)
        nonnegative=method in {'count','sum'} and all(r[1] is not None and r[1]>=0 for r in rows)
        total=sum(r[1] for r in rows) if nonnegative else None
        cumulative=0;output=[]
        for label,v,n in rows:
            share=v/total*100 if total and v is not None else None
            cumulative+=share or 0
            output.append([label,v,n,number(share),number(cumulative) if share is not None else None])
        segments={'table':{'columns':[cfg['dimension'],'Value','Records','Share %','Cumulative %'],'data':output[:cfg['top_n']]},
                  'missing_label':str(labels[f[cfg['dimension']].isna()].iloc[0]) if f[cfg['dimension']].isna().any() else None,
                  'total_groups':len(rows),'shown':min(len(rows),cfg['top_n']),'pareto_available':bool(nonnegative and total and total>0),
                  'note':'Shares use all segment totals, including groups outside the displayed top N. Share/Pareto is available only for nonnegative additive totals or record counts.'}
        if cfg['pivot']:
            a,b=labels,_label_series(f[cfg['pivot']])
            if a.nunique()>100 or b.nunique()>30:raise ValueError('Pivot supports at most 100 row groups and 30 heading groups.')
            cells={};row_names=list(dict.fromkeys(a));col_names=list(dict.fromkeys(b))
            for (left,right),part in f.groupby([a,b],observed=True,sort=False):cells[left,right]=aggregate(part,measure,method)
            pivot={'columns':[cfg['dimension'],*col_names],
                   'data':[[label,*[cells.get((label,c),0 if method=='count' else None) for c in col_names]] for label in row_names]}
    if dates is not None:
        selected=dates.iloc[positions];valid=selected.notna()
        periods=selected.dt.tz_localize(None).dt.to_period(GRAINS[cfg['grain']])
        rows=[]
        for period,part in f.loc[valid.to_numpy()].groupby(periods.loc[valid].astype(str),sort=True):
            rows.append([str(period),aggregate(part,measure,method),len(part)])
        if len(rows)>1000:raise ValueError('Trend exceeds 1,000 periods. Choose a wider time grain or date range.')
        current,previous=(rows[-1][1],rows[-2][1]) if len(rows)>=2 else (None,None)
        ordinals={str(p):int(p.ordinal) for p in periods.dropna().unique()}
        trend={'table':{'columns':['Period','Value','Records'],'data':rows},'invalid_dates':int((~valid).sum()),
               'positions':[ordinals[row[0]] for row in rows],
               'change_percent':_percent(current,previous),'current_period':rows[-1][0] if rows else None,
               'previous_period':rows[-2][0] if len(rows)>=2 else None,
               'note':'UTC calendar buckets. Only observed periods appear; gaps are not treated as zero. Latest/previous observed buckets may be partial. This is descriptive change, not a forecast.'}
    if cfg['cost']:
        a,b=clean_numeric(f[measure]).astype('float64'),clean_numeric(f[cfg['cost']]).astype('float64');paired=a.notna()&b.notna()
        revenue=number(a[paired].sum()) if paired.any() else None
        cost=number(b[paired].sum()) if paired.any() else None
        profit=number(revenue-cost) if revenue is not None and cost is not None else None
        finance={'revenue':revenue,'cost':cost,'profit':profit,'paired_rows':int(paired.sum()),
                 'margin_percent':number(profit/revenue*100) if profit is not None and revenue not in {None,0} else None,
                 'note':'Profit and margin use only rows with both finite revenue and cost. Missing values are not imputed; refunds/negative amounts are retained.'}
    target=cfg['target'];scenario=number(value*(1+cfg['scenario_percent']/100)) if value is not None else None
    detail={'columns':['Active row',*list(frame.columns[:30])],
            'data':[[int(pos)+1,*[_scalar(v) for v in frame.iloc[pos,:30]]] for pos in positions[:200]]}
    return {'config':cfg,'total_rows':len(frame),'filtered_rows':len(f),'metric':metric,'segments':segments,'pivot':pivot,'trend':trend,'finance':finance,
            'distinct_customers':int(f[cfg['customer']].nunique()) if cfg['customer'] else None,
            'target':{'value':target,'gap':number(value-target) if value is not None and target is not None else None,
                      'attainment_percent':number(value/target*100) if value is not None and target is not None and target>0 else None},
            'scenario':{'percent':cfg['scenario_percent'],'value':scenario,'note':'Arithmetic sensitivity only: the selected metric changes by the entered percentage. Not a prediction or causal estimate.'},
            'quality':{'missing_cells':int(f.isna().sum().sum()),'duplicate_extras':int(f.duplicated().sum())},
            'detail':detail,'_positions':positions.tolist(),
            'notes':['All measures respect the current AND/date filters. Original records are unchanged.',
                     'Sum/mean/min/max/median exclude missing and infinite measurements; all-missing results stay undefined.',
                     'Drill-down preview shows up to 200 rows and 30 columns; filtered-data export retains every selected row and column.']}


def _scalar(value):
    if pd.isna(value):return None
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float):return value if np.isfinite(value) else str(value)
    return value if isinstance(value,(str,bool,int)) else str(value)
