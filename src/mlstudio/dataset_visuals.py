"""Customizable dataset charts with explicit outlier rules and correlation evidence."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .visual_requests import execute_request
from .preprocessing import iqr_evidence

COLORS={'purple':'#8966dc','blue':'#5b91d6','teal':'#299e91','orange':'#d79445'}


def numeric_columns(frame):
    return [c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]


def detect_outliers(frame,columns,method='iqr'):
    if method not in {'iqr','zscore'}:raise ValueError('Choose IQR or Z-score outlier detection.')
    if not columns or len(columns)>20 or any(c not in numeric_columns(frame) for c in columns):
        raise ValueError('Choose 1–20 numeric columns to inspect for outliers.')
    mask=np.zeros(len(frame),dtype=bool);fields=[]
    for c in columns:
        if method=='iqr':
            flagged,detail=iqr_evidence(frame,c)
        else:
            finite=frame[c].replace([np.inf,-np.inf],np.nan)
            mean=finite.mean();std=finite.std(ddof=0)
            lower,upper=mean-3*std,mean+3*std
            if any(pd.notna(v) and not np.isfinite(v) for v in [mean,std,lower,upper]):
                raise ValueError('Outlier statistics exceed the numeric range. Review the measurement scale.')
            flagged=(((finite-mean).abs()>3*std) & finite.notna()).fillna(False).to_numpy(dtype=bool) if pd.notna(std) and std>0 else np.zeros(len(frame),dtype=bool)
            detail={'column':c,'mean':float(mean) if pd.notna(mean) else None,
                    'std':float(std) if pd.notna(std) else None,'lower':float(lower) if pd.notna(lower) else None,
                    'upper':float(upper) if pd.notna(upper) else None,'finite_count':int(finite.notna().sum()),
                    'missing_count':int(frame[c].isna().sum()),'infinite_count':int(frame[c].isin([np.inf,-np.inf]).sum()),
                    'outlier_count':int(flagged.sum()),'method':'Absolute Z-score > 3; population standard deviation (ddof=0)'}
        mask|=flagged;fields.append(detail)
    return mask,fields


def build_chart(frame,config):
    if not isinstance(config,dict):raise ValueError('Choose your chart settings.')
    if frame.empty or len(frame)>250000:raise ValueError('Dataset visuals supports 1–250,000 active rows.')
    kind=config.get('kind','scatter')
    if kind not in {'scatter','histogram','box','bar','correlation'}:raise ValueError('Choose a chart from the available tabs.')
    columns=config.get('columns',[])
    if not isinstance(columns,list) or len(set(columns))!=len(columns) or any(c not in frame for c in columns):
        raise ValueError('Choose distinct columns from the dataset.')
    if not columns and kind!='bar':raise ValueError('Choose at least one numeric measurement.')
    color=COLORS.get(config.get('color','purple'))
    if color is None:raise ValueError('Choose one of the available chart colors.')
    opacity=config.get('opacity',.65);point_size=config.get('point_size',22)
    if isinstance(opacity,bool) or not isinstance(opacity,(int,float)) or not .1<=opacity<=1:
        raise ValueError('Opacity must be between 0.1 and 1.')
    if type(point_size) is not int or not 5<=point_size<=100:raise ValueError('Point size must be between 5 and 100.')
    titles={key:str(config.get(key,'')).strip()[:160] for key in ['title','xlabel','ylabel']}
    rule=config.get('outlier_method','iqr');show=config.get('show_outliers',True)
    if type(show) is not bool:raise ValueError('Choose whether to show outliers.')
    outlier_cols=columns if kind!='bar' else []
    if any(c not in numeric_columns(frame) for c in outlier_cols):raise ValueError('Choose numeric columns for this chart.')
    mask,fields=detect_outliers(frame,outlier_cols,rule) if outlier_cols else (np.zeros(len(frame),dtype=bool),[])
    flagged_positions=np.flatnonzero(mask)
    eligible_columns=outlier_cols[:20]
    positions=flagged_positions[:50]
    preview_values=json_values(frame.iloc[positions][eligible_columns])
    outliers={'method':rule,'count':int(mask.sum()),'fields':fields,
              'preview':{'columns':['Active row',*eligible_columns],'index':list(range(len(positions))),
                         'data':[[int(pos)+1,*row] for pos,row in zip(positions,preview_values)]},
              'note':'Outliers are unusual values, not confirmed errors. Counts use all active rows. Missing and infinite values are excluded from numeric outlier rules. At most 50 flagged rows are displayed.'}
    support=None;pair_table=None
    if kind=='correlation':
        if not 2<=len(columns)<=20:raise ValueError('Choose 2–20 numeric columns for a correlation matrix.')
        method=config.get('correlation','pearson')
        if method not in {'pearson','spearman','kendall'}:raise ValueError('Choose Pearson, Spearman or Kendall correlation.')
        if method=='kendall' and len(frame)>10000:raise ValueError('Kendall is limited to 10,000 rows. Use Pearson/Spearman or a smaller dataset.')
        clean=frame[columns].replace([np.inf,-np.inf],np.nan)
        matrix=clean.corr(method=method,min_periods=2)
        figure=Figure(figsize=(10,6),layout='constrained');axis=figure.add_subplot(111)
        image=axis.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        axis.set_xticks(range(len(columns)),columns,rotation=40,ha='right');axis.set_yticks(range(len(columns)),columns)
        figure.colorbar(image,ax=axis,label=method.title()+' correlation')
        table={'columns':['Column',*columns],'index':list(range(len(columns))),
               'data':[[c,*[finite(v) for v in matrix.loc[c]]] for c in columns]}
        valid=clean.notna().astype('int64');counts=valid.T.dot(valid)
        support={'columns':['Column',*columns],'index':list(range(len(columns))),
                 'data':[[c,*[int(v) for v in counts.loc[c]]] for c in columns]}
        pairs=[]
        for i,a in enumerate(columns):
            for b in columns[i+1:]:
                r=finite(matrix.loc[a,b])
                pairs.append([a,b,r,int(counts.loc[a,b])])
        pairs.sort(key=lambda row:abs(row[2]) if row[2] is not None else -1,reverse=True)
        pair_table={'columns':['First measurement','Second measurement',method.title()+' correlation','Finite pairs'],
                    'index':list(range(len(pairs))),'data':pairs}
        notes=[f'{method.title()} correlation uses finite pairs per cell. Pair counts are available below. Constant or insufficient columns produce blank coefficients. Correlation does not establish causation.']
        insights=[f'{a} × {b}: {r:.3f} ({n:,} finite pairs).' for a,b,r,n in pairs[:3] if r is not None]
        if not insights:insights=['No defined correlation was found. Check constant columns, missingness and sample size.']
        title=titles['title'] or method.title()+' correlation'
    else:
        plan={**titles,'kind':'grouped' if kind=='bar' else kind,'columns':columns,
              'group':config.get('group',''),'statistic':config.get('statistic','count'),'layout':'combined',
              'bins':config.get('bins',20),'orientation':config.get('orientation','vertical'),
              'top_n':config.get('top_n',20),'order':config.get('order','descending')}
        result,figure=execute_request(frame,plan)
        table=result['table'];support=result.get('support');title=result['title']
        notes=[result['protocol']];insights=result['insights'];axis=figure.axes[0]
        if kind=='scatter':
            # Select positions rather than index labels: retained indices need not be unique.
            validmask=frame[columns].replace([np.inf,-np.inf],np.nan).notna().all(axis=1).to_numpy()
            all_positions=np.flatnonzero(validmask)
            chosen=pd.Series(all_positions).sample(n=5000,random_state=42).to_numpy() if len(all_positions)>5000 else all_positions
            points=frame.iloc[chosen][columns]
            axis.clear();normal=~mask[chosen] if show else np.ones(len(chosen),dtype=bool)
            if normal.any():axis.scatter(points.iloc[normal,0],points.iloc[normal,1],s=point_size,alpha=opacity,color=color,label='Measurements' if show else None)
            flagged=mask[chosen] if show else np.zeros(len(chosen),dtype=bool)
            if flagged.any():axis.scatter(points.iloc[flagged,0],points.iloc[flagged,1],s=point_size*1.4,alpha=min(1,opacity+.2),color='#e06a75',marker='x',label='Flagged by '+rule.upper())
            if flagged.any():axis.legend()
            axis.set_xlabel(titles['xlabel'] or columns[0]);axis.set_ylabel(titles['ylabel'] or columns[1])
            notes.append(f'{len(chosen):,} complete finite pairs plotted; a deterministic sample is used above 5,000. Outliers are flagged if either axis exceeds its univariate rule, not a multivariate anomaly model.')
        elif kind=='histogram':
            for patch in axis.patches:patch.set_facecolor(color);patch.set_alpha(opacity)
            if show:
                detail=fields[0]
                for key in ['lower','upper']:
                    bound=detail[key]
                    if bound is not None:axis.axvline(bound,color='#d95468',linestyle='--',linewidth=1.3,label='Outlier limits' if key=='lower' else None)
                if detail['lower'] is not None:axis.legend()
        elif kind=='box':
            axis.clear();valid=[]
            for c in columns:
                values=frame[c].replace([np.inf,-np.inf],np.nan).dropna()
                if not values.empty:valid.append((c,values))
            axis.boxplot([s.to_numpy() for _,s in valid],tick_labels=[c for c,_ in valid],patch_artist=True,
                         showfliers=show,boxprops={'facecolor':color,'alpha':opacity},
                         flierprops={'marker':'o','markerfacecolor':'#df6c7c','markeredgecolor':'#df6c7c','markersize':4,'alpha':.65})
            axis.set_xlabel(titles['xlabel'] or 'Measurement');axis.set_ylabel(titles['ylabel'] or 'Value');axis.tick_params(axis='x',labelrotation=25)
            notes.append('Boxplot boxes and fliers always use the conventional 1.5 × IQR rule. The separately selected outlier rule affects the flagged-row table, not boxplot geometry.')
        elif kind=='bar':
            for patch in axis.patches:patch.set_facecolor(color);patch.set_alpha(opacity)
    axis.set_title(title)
    if kind=='correlation':
        if titles['xlabel']:axis.set_xlabel(titles['xlabel'])
        if titles['ylabel']:axis.set_ylabel(titles['ylabel'])
    if config.get('grid',True) and kind!='correlation':axis.grid(axis='y',alpha=.17);axis.set_axisbelow(True)
    interactive=None
    if kind=='scatter':
        interactive={'columns':columns,'points':[[int(pos),float(frame.iloc[pos][columns[0]]),float(frame.iloc[pos][columns[1]])] for pos in chosen],
            'note':'Up to 5,000 finite points shown. Region selection checks all active rows, including points omitted from this display.'}
    return {'title':title,'kind':kind,'config':config,'table':table,'support':support,'pairs':pair_table,'interactive':interactive,
            'insights':insights,'outliers':outliers,'rows':len(frame),
            'protocol':' '.join(notes)+' All chart operations are exploratory and leave the active dataset unchanged.'},figure


def finite(v):
    return float(v) if pd.notna(v) and np.isfinite(v) else None


def select_region(frame,config):
    columns=config.get('columns',[])
    if len(columns)!=2 or any(c not in numeric_columns(frame) for c in columns):raise ValueError('Choose two numeric scatter axes.')
    bounds=[config.get(k) for k in ['xmin','xmax','ymin','ymax']]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in bounds):raise ValueError('Select a finite plot region.')
    xmin,xmax,ymin,ymax=bounds
    if xmin>xmax or ymin>ymax:raise ValueError('Invalid selection bounds.')
    x,y=frame[columns[0]],frame[columns[1]]
    mask=(x.between(xmin,xmax)&y.between(ymin,ymax)).fillna(False).to_numpy(dtype=bool)
    positions=np.flatnonzero(mask).tolist();subset=frame.iloc[positions[:100]]
    return {'positions':positions,'count':len(positions),'bounds':dict(zip(['xmin','xmax','ymin','ymax'],bounds)),
        'preview':{'columns':['Active row',*frame.columns],'index':list(range(len(subset))),
                   'data':[[pos+1,*row] for pos,row in zip(positions,json_values(subset))]}}


def json_values(frame):
    rows=[]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for v in row:
            if pd.isna(v):values.append(None)
            elif isinstance(v,(float,np.floating)) and not np.isfinite(v):values.append('Infinity' if v>0 else '-Infinity')
            elif isinstance(v,(int,np.integer)):values.append(int(v))
            elif isinstance(v,(float,np.floating)):values.append(float(v))
            else:values.append(str(v))
        rows.append(values)
    return rows
