"""Transparent offline request matching and bounded data visualization, without code evaluation."""
import re
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .lab import grouped_analysis

KINDS={"grouped":"Grouped bar chart","histogram":"Histogram","scatter":"Scatter plot",
       "box":"Box plot","correlation":"Correlation heatmap","missing":"Missing-value chart",
       "line":"Trend / line chart","category_share":"Category share (%)"}
STATS={"mean":r"average|avg|mean|গড়|গড়|gor|goro", "median":r"median|মধ্যক",
       "sum":r"sum|total|মোট|mot", "min":r"minimum|min|সর্বনিম্ন",
       "max":r"maximum|max|সর্বোচ্চ", "std":r"standard deviation|std|spread",
       "count":r"count|frequency|frequencies|number of|কতটি|কতগুলো|সংখ্যা|koyta|kotogulo"}


def normalized(text):
    return re.sub(r"[^\w\u0980-\u09ff]+", " ", str(text).lower().replace("_"," ")).strip()


def matches(text, pattern):
    return re.search(r"(?<!\w)(?:"+pattern+r")(?!\w)",text,re.I) is not None


def parse_request(frame, question):
    if not isinstance(question,str) or not question.strip() or len(question)>4000:
        raise ValueError("Write one requirement, up to 4,000 characters.")
    text=normalized(question)
    found=[];positions={}
    for column in frame:
        aliases={normalized(column),normalized(re.sub(r"\([^)]*\)","",str(column)))}
        locations=[m.start() for alias in aliases if alias for m in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',text)]
        if locations:found.append(column);positions[column]=min(locations)
    numeric=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    categories=[c for c in frame if c not in numeric and frame[c].nunique(dropna=False)<=30]
    plan={"kind":"grouped","group":"","columns":[],"statistic":"mean","layout":"combined",
          "orientation":"horizontal" if matches(text,"horizontal|অনুভূমিক") else "vertical",
          "top_n":None,"order":"original","bins":20,"title":"","xlabel":"","ylabel":"",
          "filter_column":"","filter_operator":"none","filter_value":""}
    plan.update(x_column="",series_column="")
    questions=[];assumptions=[]
    intents=[p for p in ["scatter|স্ক্যাটার", "correlation|heatmap|সহসম্পর্ক", "histogram|হিস্টোগ্রাম", "box plot|boxplot|বক্স", "missing|মিসিং", "line chart|trend|time series|ধারা", "percentage|percent|proportion|share|শতাংশ"] if matches(text,p)]
    if len(intents)>1:questions.append("Several chart types were mentioned. Choose one chart per result and add the next request separately.")
    if matches(text,"line chart|line plot|trend|time series|ধারা"):plan['kind']='line'
    elif matches(text,"percentage|percent|proportion|share|শতাংশ"):plan['kind']='category_share';plan['statistic']='count'
    elif matches(text,"scatter|relationship|versus|vs|স্ক্যাটার|সম্পর্ক"):plan['kind']='scatter'
    elif matches(text,"correlation|heatmap|সহসম্পর্ক"):plan['kind']='correlation'
    elif matches(text,"histogram|distribution|হিস্টোগ্রাম|বণ্টন"):plan['kind']='histogram'
    elif matches(text,"box plot|boxplot|বক্স"):plan['kind']='box'
    elif matches(text,"missing|মিসিং|অনুপস্থিত"):plan['kind']='missing'
    elif not (any(matches(text,p) for p in STATS.values()) or matches(text,"bar|বার")):
        questions.append("Choose a supported chart or calculation in the plan below.")
    if matches(text,"pie|donut|regression|prediction|train|predict|animation"):
        questions.append("This request includes a chart/model type outside this visual request builder. Choose one supported chart; use Model Lab for training.")
    if matches(text,"where|filter|only|যেখানে|শুধু") or re.search(r'[<>=]',question):
        questions.append("Set the row filter below explicitly; the written condition has not been applied.")
    statistics=[key for key,pattern in STATS.items() if matches(text,pattern)]
    # 'number of features' and 'total numerical features' describe selections, not row count/sum.
    if matches(text,"average|avg|mean|গড়|গড়"):
        statistics=[s for s in statistics if s not in {'count','sum'}]
    if len(statistics)>1:questions.append("Choose one calculation per result; multiple calculations were mentioned.")
    if statistics:plan['statistic']=statistics[0]
    if matches(text,"subplot|subplots|vertically|separate.*group|প্রতিটি.*আলাদা"):
        plan['layout']='subplots';plan['orientation']='vertical'
    top=re.search(r'\btop\s+(\d+)\b',text)
    if top:
        plan['top_n']=int(top.group(1));plan['order']='descending'
        if not 1<=plan['top_n']<=30:questions.append("Choose a top-group count between 1 and 30.")
    if matches(text,"ascending|smallest|lowest|ছোট থেকে"):plan['order']='ascending'
    elif matches(text,"descending|largest|highest|বড় থেকে"):plan['order']='descending'
    bins=re.search(r'\b(\d+)\s+bins?\b',text)
    if bins:plan['bins']=int(bins.group(1))
    if plan['kind']=='line':
        explicit_x=[];explicit_series=[]
        for c in found:
            alias=normalized(c)
            if re.search(r'(?:over|against|along|x axis)\s+(?:the\s+)?'+re.escape(alias)+r'(?!\w)',text):explicit_x.append(c)
            if re.search(r'(?:by|per|each|series|color)\s+(?:the\s+)?'+re.escape(alias)+r'(?!\w)',text):explicit_series.append(c)
        nonnumeric=[c for c in found if c not in numeric]
        date_candidates=[c for c in nonnumeric if pd.api.types.is_datetime64_any_dtype(frame[c]) or
                         (frame[c].notna().any() and frame[c].dropna().astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all())]
        if len(explicit_x)==1:plan['x_column']=explicit_x[0]
        elif not explicit_x and len(date_candidates)==1:
            plan['x_column']=date_candidates[0];assumptions.append(f"Used {date_candidates[0]} as the time axis; review the choice.")
        else:questions.append("Choose the numeric or date column for the X axis. Use 'over COLUMN' to name it explicitly.")
        candidates=[c for c in explicit_series if c!=plan['x_column']]
        if len(candidates)==1:plan['series_column']=candidates[0]
        elif len(candidates)>1:questions.append("Choose at most one separate-series column.")
        plan['columns']=sorted([c for c in found if c in numeric and c not in {plan['x_column'],plan['series_column']}],key=lambda c:positions[c])
        if not plan['columns']:questions.append("Choose one or more numeric measurements for the Y axis.")
        if len(plan['columns'])>4:questions.append("Choose at most four Y measurements for a readable line chart.")
        assumptions.append("Repeated X values are aggregated using the selected calculation; points are sorted on X. Separate series are never connected to each other.")
        if not statistics:assumptions.append("Selected mean to combine repeated X values. Change the calculation if a total is appropriate.")
    elif plan['kind'] in {'grouped','category_share'}:
        if not statistics and plan['kind']=='grouped':assumptions.append("No calculation was named; selected average (mean). Change it in the plan if needed.")
        # Prefer a column explicitly following a grouping phrase, even if it is numeric.
        explicit=[]
        for c in found:
            alias=normalized(c)
            if re.search(r'(?:by|per|each|অনুযায়ী|অনুযায়ী)\s+(?:the\s+)?'+re.escape(alias)+r'(?!\w)',text):explicit.append(c)
        groups=explicit or [c for c in found if c not in numeric]
        if len(groups)==1:plan['group']=groups[0]
        elif len(groups)>1:questions.append("More than one grouping column was mentioned. Choose the grouping column.")
        elif len(categories)==1 and not matches(text,"by|per|each|প্রতি"):
            plan['group']=categories[0];assumptions.append(f"Used {categories[0]} because it is the only category column with at most 30 groups.")
        else:questions.append("Choose the column that defines your groups.")
        columns=[c for c in found if c in numeric and c!=plan['group']]
        if matches(text,"all numeric|all numerical|four numerical|four numeric|সব.*সংখ্যা|সব numeric|all features"):
            columns=[c for c in numeric if c!=plan['group']]
            assumptions.append("Selected all numeric columns except the grouping column; review IDs and outcome columns.")
        plan['columns']=columns
        if plan['kind']=='category_share':plan['statistic']='count';plan['columns']=[]
        if plan['statistic']!='count' and not columns:questions.append("Choose the numeric measurement(s) to summarize.")
        if len(columns)>12:questions.append("Select at most 12 numeric measurements.")
    elif plan['kind']=='missing':
        plan['columns']=found or list(frame.columns[:40])
        if not found and len(frame.columns)>40:assumptions.append("Selected the first 40 columns for missingness; choose other columns in the plan if needed.")
    else:
        plan['columns']=[c for c in found if c in numeric]
        need=2 if plan['kind'] in {'scatter','correlation'} else 1
        if plan['kind']=='correlation' and not plan['columns']:
            plan['columns']=numeric[:20];assumptions.append("Used up to 20 numeric columns for the correlation matrix.")
        if len(plan['columns'])<need:questions.append(f"Choose {'at least two numeric columns' if need==2 else 'at least one numeric column'}.")
        if plan['kind']=='scatter':
            # Plot order follows first textual mention, rather than CSV order.
            plan['columns'].sort(key=lambda c:positions[c])
            if len(plan['columns'])!=2:questions.append("Scatter plots need exactly two selected columns.")
        if plan['kind']=='histogram' and len(plan['columns'])!=1:questions.append("Choose one numeric column for the histogram.")
    if plan['kind']=='missing' and matches(text,'by|per|each|প্রতি'):
        questions.append("Missingness by group needs an explicit plan: choose a grouped row count and set an Is missing filter on the measurement.")
    return {"question":question.strip(),"plan":plan,"questions":list(dict.fromkeys(questions)),
            "assumptions":assumptions,"ready":not questions,
            "method":"Offline keyword and dataset-column matching. Review and edit this plan before calculating."}


def parse_requests(frame, text):
    """Prepare a bounded list of independent requirements; never execute on parsing."""
    if not isinstance(text,str) or not text.strip() or len(text)>12000:
        raise ValueError("Write up to six requirements, at most 12,000 characters in total.")
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    numbered=re.compile(r'^\s*(?:(?:problem|question|task)\s*\d+\s*[:.)-]|\d+\s*[.)]|[-•])\s*',re.I)
    chunks=[]
    if any(numbered.match(line) for line in lines):
        for line in lines:
            if numbered.match(line):chunks.append(numbered.sub('',line,count=1))
            elif chunks:chunks[-1]+=' '+line
            else:chunks.append(line)
    else:chunks=lines
    if not 1<=len(chunks)<=6:raise ValueError("Use one requirement per line, up to six requirements.")
    return [parse_request(frame,q) for q in chunks]


def _table(columns,rows):
    return {"columns":list(columns),"index":list(range(len(rows))),"data":rows}


def _number(value):
    try:return float(value) if np.isfinite(value) else None
    except (TypeError,ValueError):return None


def _line_summary(filtered,plan,columns,xlabel,ylabel):
    x_column=plan.get('x_column');series_column=plan.get('series_column','')
    if x_column not in filtered or x_column in columns:
        raise ValueError("Choose a separate X-axis column and numeric Y measurements.")
    if not 1<=len(columns)<=4 or any(not pd.api.types.is_numeric_dtype(filtered[c]) or pd.api.types.is_bool_dtype(filtered[c]) for c in columns):
        raise ValueError("Choose 1–4 numeric Y measurements for the line chart.")
    if series_column and (series_column not in filtered or series_column==x_column or series_column in columns):
        raise ValueError("Choose a separate category column for series, or leave it empty.")
    stat=plan.get('statistic','mean')
    if stat not in {'mean','median','sum','min','max'}:
        raise ValueError("Choose mean, median, sum, minimum or maximum for repeated X values.")
    x=filtered[x_column]
    dates=pd.api.types.is_datetime64_any_dtype(x)
    if dates:pass
    elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):x=x.replace([np.inf,-np.inf],np.nan)
    else:
        observed=x.dropna().astype(str)
        if observed.empty or not observed.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError("The X axis must be numeric, a datetime column, or unambiguous ISO dates (YYYY-MM-DD). Correct date types in Preprocessing first.")
        try:x=pd.to_datetime(x,format='%Y-%m-%d',errors='raise')
        except (ValueError,TypeError):raise ValueError("Some X-axis dates are invalid. Correct them in Preprocessing first.")
        dates=True
    # Internal names avoid collisions with arbitrary user column names.
    clean=pd.DataFrame({'x':x.to_numpy()},index=filtered.index)
    raw_series=filtered[series_column] if series_column else pd.Series('All rows',index=filtered.index)
    missing_label='(missing series)';observed_labels=set(raw_series.dropna().astype(str))
    while missing_label in observed_labels:missing_label='_'+missing_label
    clean['series']=raw_series.map(lambda v:missing_label if pd.isna(v) else str(v))
    if clean['series'].nunique()*len(columns)>12:
        raise ValueError("The line chart supports at most 12 measurement/series combinations. Filter the data or choose fewer measurements.")
    for i,c in enumerate(columns):clean['y'+str(i)]=filtered[c].replace([np.inf,-np.inf],np.nan).to_numpy()
    eligible=clean.loc[clean.x.notna()]
    if eligible.empty:raise ValueError("No valid X-axis values remain after filtering.")
    value_columns=['y'+str(i) for i in range(len(columns))]
    groups=eligible.groupby(['series','x'],sort=True,dropna=False)
    summary=groups[value_columns].sum(min_count=1) if stat=='sum' else groups[value_columns].agg(stat)
    counts=groups[value_columns].count()
    if len(summary)>5000:
        raise ValueError("More than 5,000 X/series points remain. Add a row filter or aggregate your dates before plotting.")
    if not summary.notna().any(axis=None):raise ValueError("No finite Y measurements remain to plot.")
    figure=Figure(figsize=(10,5),layout='constrained');axis=figure.add_subplot(111)
    rows=[];support=[];notes=[];insights=[]
    for series in summary.index.get_level_values(0).unique():
        part=summary.xs(series,level=0).sort_index()
        for i,c in enumerate(columns):
            values=part['y'+str(i)].to_numpy(dtype=float)
            axis.plot(part.index,values,marker='o',markersize=3,linewidth=1.7,
                      label=(series+' · ' if series_column else '')+c)
            valid=part['y'+str(i)].dropna()
            if len(valid)>1:
                insights.append(f"{series+' · ' if series_column else ''}{c}: first plotted aggregate {valid.iloc[0]:.4g}, last {valid.iloc[-1]:.4g}; {len(valid)} X positions with recorded values.")
        for index,values in part.iterrows():
            x_value=index.isoformat() if dates else _number(index)
            prefix=[series,x_value] if series_column else [x_value]
            rows.append([*prefix,*[_number(v) for v in values]])
            support.append([*prefix,*[int(v) for v in counts.loc[(series,index)].tolist()]])
    axis.set_xlabel(xlabel or x_column);axis.set_ylabel(ylabel or (columns[0] if len(columns)==1 else 'Value'))
    axis.legend();axis.tick_params(axis='x',labelrotation=25)
    prefix=[series_column,x_column] if series_column else [x_column]
    table=_table([*prefix,*columns],rows)
    support_table=_table([*prefix,*[c+' · finite values' for c in columns]],support)
    notes.append(f"{len(eligible):,} rows have a valid X value. Duplicate X values are aggregated using {stat} within each separate series; every Y uses its own finite-value count. Points are sorted on X; series are not connected to one another.")
    if dates:notes.append("Dates are existing datetime values or strict YYYY-MM-DD strings; no day/month guessing is performed.")
    insights.append("A connected line describes observed values or aggregates; it is not a forecast or proof of a causal trend.")
    return table,support_table,figure,insights,notes


def _category_share(filtered,plan,xlabel,ylabel):
    group=plan.get('group')
    if group not in filtered:raise ValueError("Choose a category column for percentage shares.")
    counts=filtered.groupby(group,dropna=False,observed=True,sort=False).size()
    total_groups=len(counts);top_n=plan.get('top_n')
    if top_n is not None and (type(top_n) is not int or not 1<=top_n<=30):raise ValueError("Choose 1–30 top groups.")
    if total_groups>30 and top_n is None:raise ValueError("More than 30 groups. Choose top groups or a different category column.")
    order=plan.get('order','original')
    if order not in {'original','ascending','descending'}:raise ValueError("Choose a supported sort order.")
    if order!='original':counts=counts.sort_values(ascending=order=='ascending',kind='stable')
    if top_n is not None:counts=counts.head(top_n)
    missing_label='(missing group)';existing={str(v) for v in counts.index if not pd.isna(v)}
    while missing_label in existing:missing_label='_'+missing_label
    labels=[missing_label if pd.isna(v) else str(v) for v in counts.index]
    percentages=counts.to_numpy(dtype=float)/len(filtered)*100
    figure=Figure(figsize=(10,5),layout='constrained');axis=figure.add_subplot(111)
    orientation=plan.get('orientation','vertical')
    if orientation=='horizontal':
        axis.barh(labels,percentages,color='#8966dc');axis.invert_yaxis()
        axis.set_xlabel(xlabel or 'Percent of filtered rows');axis.set_ylabel(ylabel or group)
    elif orientation=='vertical':
        axis.bar(labels,percentages,color='#8966dc');axis.tick_params(axis='x',labelrotation=25)
        axis.set_xlabel(xlabel or group);axis.set_ylabel(ylabel or 'Percent of filtered rows')
    else:raise ValueError("Choose vertical or horizontal bars.")
    rows=[[label,int(count),float(percent)] for label,count,percent in zip(labels,counts,percentages)]
    table=_table([group,'Rows','Percent of filtered rows'],rows)
    notes=[f"Showing {len(counts)} of {total_groups} groups. Percentage denominator is all {len(filtered):,} filtered rows, including missing category labels. Top-N does not renormalize the displayed percentages."]
    largest=int(np.argmax(percentages))
    insights=[f"Largest displayed share: {labels[largest]} with {int(counts.iloc[largest]):,} rows ({percentages[largest]:.2f}%).",
              f"Displayed groups account for {percentages.sum():.2f}% of filtered rows. Percentages describe category frequency, not model accuracy."]
    return table,figure,insights,notes


def execute_request(frame, plan):
    if not isinstance(plan,dict):raise ValueError("Review a visual plan first.")
    kind=plan.get('kind');columns=plan.get('columns',[])
    if kind not in KINDS:raise ValueError("Choose a supported visualization.")
    if not isinstance(columns,list) or not all(isinstance(c,str) and c in frame for c in columns) or len(set(columns))!=len(columns):
        raise ValueError("Choose distinct columns from the active dataset.")
    if len(frame)>250000:raise ValueError("Visual requests support up to 250,000 rows. Use a smaller dataset.")
    if frame.empty:raise ValueError("Import a dataset with rows first.")
    total=len(frame);filtered=frame
    op=plan.get('filter_operator','none');filter_note='No row filter.'
    if op!='none':
        column=plan.get('filter_column');value=plan.get('filter_value','')
        if column not in frame:raise ValueError("Choose a valid filter column.")
        s=frame[column]
        if op in {'gt','ge','lt','le'}:
            if not pd.api.types.is_numeric_dtype(s):raise ValueError("Numeric comparisons need a numeric filter column.")
            try:threshold=float(value)
            except (ValueError,TypeError):raise ValueError("Enter a numeric filter value.")
            if not np.isfinite(threshold):raise ValueError("Use a finite filter value.")
            mask={'gt':s.gt,'ge':s.ge,'lt':s.lt,'le':s.le}[op](threshold)
        elif op in {'eq','ne'}:
            if pd.api.types.is_numeric_dtype(s):
                try:target=float(value)
                except (TypeError,ValueError):raise ValueError("Enter a numeric filter value.")
                if not np.isfinite(target):raise ValueError("Use a finite filter value.")
                mask=s.eq(target)
            else:mask=s.astype('string').eq(str(value))
            if op=='ne':mask=~mask
        elif op=='contains':mask=s.astype('string').str.contains(str(value),case=False,regex=False,na=False)
        elif op=='missing':mask=s.isna()
        elif op=='present':mask=s.notna()
        else:raise ValueError("Choose a supported filter operation.")
        if op not in {'missing','present'}:mask=mask & s.notna()
        filtered=frame.loc[mask.fillna(False)]
        filter_note=f"Filter: {column} {op} {value if op not in {'missing','present'} else ''}."
        if filtered.empty:raise ValueError("No rows match this filter. Change or clear the filter.")
    def bounded_text(key,default):return str(plan.get(key,'')).strip()[:160] or default
    title=bounded_text('title',KINDS[kind])
    xlabel=bounded_text('xlabel','');ylabel=bounded_text('ylabel','')
    insights=[];notes=[];support_table=None
    if kind=='line':
        table,support_table,figure,insights,notes=_line_summary(filtered,plan,columns,xlabel,ylabel)
        figure.axes[0].set_title(title)
    elif kind=='category_share':
        table,figure,insights,notes=_category_share(filtered,plan,xlabel,ylabel)
        figure.axes[0].set_title(title)
    elif kind=='grouped':
        result,figure=grouped_analysis(filtered,plan.get('group'),columns,plan.get('statistic','mean'),
            plan.get('layout','combined'),title=plan.get('title',''),xlabel=xlabel,ylabel=ylabel,
            top_n=plan.get('top_n'),order=plan.get('order','original'),orientation=plan.get('orientation','vertical'))
        title=result['title'];table=result['table'];notes.append(result['note'])
        group=plan.get('group')
        observed=filtered.groupby(group,dropna=False,observed=True,sort=False)
        label="(missing group)";existing={str(v) for v in observed.size().index if not pd.isna(v)}
        while label in existing:label='_'+label
        support={}
        for key,part in observed:
            name=label if pd.isna(key) else str(key)
            support[name]=[name,len(part),*[int(part[c].replace([np.inf,-np.inf],np.nan).notna().sum()) for c in columns]]
        support_table=_table(['Group','Rows',*[c+' · finite values' for c in columns]],
                             [support[row[0]] for row in table['data']])
        for j,col in enumerate(table['columns'][1:],1):
            usable=[row for row in table['data'] if row[j] is not None]
            if usable:
                hi=max(usable,key=lambda r:r[j]);lo=min(usable,key=lambda r:r[j])
                ties=sum(row[j]==hi[j] for row in usable)
                insights.append(f"{col}: highest displayed value {hi[j]:.4g} in {hi[0]}"+(f" (tied across {ties} groups)" if ties>1 else "")+f"; lowest {lo[j]:.4g} in {lo[0]}.")
        insights.append("Groups with missing or few measurements may not be directly comparable. Group averages do not establish causation.")
    else:
        if not columns:raise ValueError("Select at least one column.")
        if len(columns)>40:raise ValueError("Select at most 40 columns.")
        if kind!='missing' and any(not pd.api.types.is_numeric_dtype(filtered[c]) or pd.api.types.is_bool_dtype(filtered[c]) for c in columns):
            raise ValueError("Select numeric measurements for this chart.")
        figure=Figure(figsize=(10,5),layout='constrained');axis=figure.add_subplot(111)
        clean=filtered[columns].replace([np.inf,-np.inf],np.nan) if kind!='missing' else filtered[columns]
        if kind=='histogram':
            if len(columns)!=1:raise ValueError("A histogram needs exactly one numeric column.")
            bins=plan.get('bins',20)
            if type(bins) is not int or not 2<=bins<=100:raise ValueError("Choose 2–100 histogram bins.")
            s=clean.iloc[:,0].dropna()
            if s.empty:raise ValueError("No finite values in this column.")
            counts,edges=np.histogram(s,bins=bins);axis.hist(s,bins=edges,color='#8966dc',edgecolor='#ffffff',linewidth=.4)
            table=_table(['Bin start','Bin end','Count'],[[float(edges[i]),float(edges[i+1]),int(v)] for i,v in enumerate(counts)])
            axis.set_xlabel(xlabel or columns[0]);axis.set_ylabel(ylabel or 'Number of rows')
            insights.append(f"{len(s):,} finite values; mean {s.mean():.4g}, median {s.median():.4g}, range {s.min():.4g}–{s.max():.4g}.")
            notes.append("Bin intervals include the left edge and exclude the right edge, except the final bin includes both.")
        elif kind=='scatter':
            if len(columns)!=2:raise ValueError("Select exactly two columns in X, Y order.")
            pairs=clean.dropna()
            if pairs.empty:raise ValueError("No rows have finite values for both columns.")
            display=pairs.sample(n=5000,random_state=42) if len(pairs)>5000 else pairs
            axis.scatter(display.iloc[:,0],display.iloc[:,1],color='#8966dc',alpha=.6,s=16)
            axis.set_xlabel(xlabel or columns[0]);axis.set_ylabel(ylabel or columns[1])
            corr=_number(pairs.iloc[:,0].corr(pairs.iloc[:,1])) if len(pairs)>1 else None
            table=_table(['Measure','Value'],[['Complete finite pairs',len(pairs)],['Pearson correlation',corr],['Plotted points',len(display)]])
            insights.append(f"{len(pairs):,} complete pairs. Pearson correlation: {corr:.4f}. Association does not establish causation." if corr is not None else "Correlation is undefined for constant columns or insufficient pairs.")
            notes.append(f"Chart shows {len(display):,} {'deterministically sampled' if len(display)<len(pairs) else 'total'} complete pairs; correlation uses all complete pairs.")
        elif kind=='box':
            usable=[c for c in columns if clean[c].notna().any()]
            if not usable:raise ValueError("No finite values for a box plot.")
            axis.boxplot([clean[c].dropna().to_numpy() for c in usable],tick_labels=usable)
            axis.tick_params(axis='x',labelrotation=25);axis.set_xlabel(xlabel or 'Measurement');axis.set_ylabel(ylabel or 'Value')
            rows=[]
            for c in usable:
                s=clean[c].dropna();q1,q3=s.quantile([.25,.75]);iqr=q3-q1
                out=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
                rows.append([c,len(s),_number(q1),_number(s.median()),_number(q3),out])
            table=_table(['Column','Finite count','25th percentile','Median','75th percentile','IQR outlier count'],rows)
            insights.append("Box boundaries mark the 25th and 75th percentiles. Points beyond 1.5 × IQR fences are potential outliers, not confirmed errors.")
        elif kind=='correlation':
            if not 2<=len(columns)<=20:raise ValueError("Select 2–20 numeric columns for correlation.")
            corr=clean.corr();image=axis.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
            axis.set_xticks(range(len(columns)),columns,rotation=45,ha='right');axis.set_yticks(range(len(columns)),columns)
            figure.colorbar(image,ax=axis,label='Pearson correlation')
            table=_table(['Column',*columns],[[c,*[_number(v) for v in corr.loc[c]]] for c in columns])
            insights.append("Pearson correlation uses finite pairs per column pair; different cells may use different row counts. Constant columns produce blank cells. Association does not establish causation.")
        else:
            counts=clean.isna().sum();axis.barh(columns,counts.to_numpy(),color='#df9b50');axis.invert_yaxis()
            axis.set_xlabel(xlabel or 'Missing cells');axis.set_ylabel(ylabel or 'Column')
            table=_table(['Column','Missing cells','Percent of rows'],[[c,int(counts[c]),float(counts[c]/len(filtered)*100)] for c in columns])
            insights.append(f"{int(counts.sum()):,} missing cells across {len(columns)} selected columns; {int(clean.isna().any(axis=1).sum()):,} rows have at least one missing entry.")
            notes.append("Missingness uses pandas isna; infinity and nonempty strings are not counted as missing.")
        axis.set_title(title)
        if kind!='missing':notes.append("Missing and infinite numeric measurements are excluded; no values are imputed.")
    protocol=f"{len(filtered):,} of {total:,} active rows after filtering. {filter_note} "+' '.join(notes)
    return {"title":title,"table":table,"support":support_table,"insights":insights,"protocol":protocol,"plan":plan,
            "rows_total":total,"rows_used":len(filtered),"columns_used":columns},figure
