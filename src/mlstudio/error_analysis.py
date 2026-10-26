"""Held-out prediction inspection with transparent subgroup sizes."""
import numpy as np
import pandas as pd


def error_report(frame, positions, actual, predicted, task):
    positions=list(map(int,positions))
    if len(positions)!=len(actual) or len(actual)!=len(predicted):
        raise ValueError("Prediction rows and labels must have matching lengths.")
    subset=frame.iloc[positions].reset_index(drop=True)
    actual=np.asarray(actual); predicted=np.asarray(predicted)
    errors=(actual!=predicted) if task=="classification" else np.abs(actual.astype(float)-predicted.astype(float))
    scores=errors.astype(float)
    def native(v):
        if isinstance(v,np.generic):v=v.item()
        return None if pd.isna(v) else v if isinstance(v,(str,int,float,bool)) else str(v)
    rows=[]
    for i,pos in enumerate(positions):
        row={"position":pos,"csv_row":pos+2,"actual":native(actual[i]),"predicted":native(predicted[i]),"features":{c:native(subset.iloc[i][c]) for c in subset},"error":bool(errors[i]) if task=="classification" else None,"absolute_error":float(errors[i]) if task=="regression" else None}
        rows.append(row)
    if task=="regression":rows.sort(key=lambda r:r["absolute_error"],reverse=True)
    else:rows.sort(key=lambda r:not r["error"])
    def summarize(labels):
        labels=pd.Series(list(labels)).astype(str)
        result=[]
        for label in labels.unique():
            mask=(labels==label).to_numpy(); count=int(mask.sum())
            result.append({"label":str(label),"count":count,"error_rate" if task=="classification" else "mae":float(scores[mask].mean()),"small_group":count<20})
        return result
    missing=subset.isna().any(axis=1).map({True:"Has missing features",False:"No missing features"})
    slices={"missingness":summarize(missing)}
    if task=="classification":slices["actual_class"]=summarize(actual)
    numeric=[c for c in subset.select_dtypes(include="number") if subset[c].nunique()>1]
    for c in numeric[:5]:
        values=subset[c]
        try:
            labels=pd.qcut(values,q=min(4,values.nunique()),duplicates="drop").astype(str).where(values.notna(),"Missing")
            slices[c]=summarize(labels)
        except ValueError:pass
    for c in [c for c in subset if c not in numeric and subset[c].nunique()<=12][:5]:
        slices[c]=summarize(subset[c].map(lambda v:"Missing" if pd.isna(v) else str(v)))
    return {"task":task,"total":len(rows),"error_count":int(scores.sum()) if task=="classification" else None,"error_rate":float(scores.mean()) if task=="classification" else None,"mae":float(scores.mean()) if task=="regression" else None,"rows":rows[:1000],"row_limit":1000,"slices":slices,"caution":"Slices describe this final test sample only. Groups with fewer than 20 rows are marked small; differences may be noise and do not prove bias or causation. Revising models after inspecting these errors makes this test set exploratory.","row_note":"Positions refer to the active dataset version (zero-based); CSV row is the row in the exported active dataset including its header, not necessarily the original source file."}
