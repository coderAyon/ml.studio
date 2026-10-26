"""Bounded, reproducible exploratory clustering with inspectable outputs."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def study(frame, columns, algorithm="K-Means", clusters=3, eps=.5, min_samples=5, scale=True, compare=False):
    if not isinstance(columns,list) or not columns or len(set(columns))!=len(columns) or any(c not in frame for c in columns):
        raise ValueError("Choose one or more different numeric feature columns.")
    if not 3<=len(frame)<=10000 or len(columns)>100:
        raise ValueError("Clustering supports 3–10,000 rows and at most 100 features.")
    numeric=frame[columns]
    if any(not pd.api.types.is_numeric_dtype(numeric[c]) or pd.api.types.is_bool_dtype(numeric[c]) for c in columns):
        raise ValueError("Choose numeric measurements; text and Boolean columns are not supported here.")
    if any(not numeric[c].notna().any() for c in columns):
        raise ValueError("Remove columns that contain only missing values.")
    if not np.isfinite(numeric.to_numpy(dtype=float,na_value=np.nan)[numeric.notna().to_numpy()]).all():
        raise ValueError("Infinite feature values must be corrected before clustering.")
    if isinstance(clusters,bool) or isinstance(min_samples,bool) or float(clusters)!=int(clusters) or float(min_samples)!=int(min_samples):raise ValueError("Group count and minimum nearby rows must be whole numbers.")
    clusters=int(clusters);min_samples=int(min_samples);eps=float(eps)
    if not 2<=clusters<=20 or ((algorithm!="DBSCAN" or compare) and clusters>=len(frame)):raise ValueError("Number of groups must be 2–20 and smaller than the row count.")
    if not np.isfinite(eps) or eps<=0 or min_samples<2 or ((algorithm=="DBSCAN" or compare) and min_samples>len(frame)):raise ValueError("DBSCAN needs a positive radius and minimum samples between 2 and the row count.")
    values=SimpleImputer(strategy="median").fit_transform(numeric)
    if scale:values=StandardScaler().fit_transform(values)
    if np.unique(values,axis=0).shape[0]<2:raise ValueError("Selected features contain no differences to group.")
    if (algorithm in {"K-Means","Hierarchical"} or compare) and clusters>np.unique(values,axis=0).shape[0]:raise ValueError("Use no more groups than the number of different measurement rows.")
    # DBSCAN can allocate quadratic neighborhoods; constrain density exploration.
    if (algorithm in {"DBSCAN","Hierarchical"} or compare) and len(frame)>3000:raise ValueError("DBSCAN, Hierarchical and comparison support up to 3,000 rows; use K-Means or a smaller extract.")
    def run(name,k=clusters):
        factories={"K-Means":lambda:KMeans(n_clusters=k,random_state=42,n_init=10),"DBSCAN":lambda:DBSCAN(eps=eps,min_samples=min_samples),"Hierarchical":lambda:AgglomerativeClustering(n_clusters=k)}
        if name not in factories:raise ValueError("Choose K-Means, DBSCAN or Hierarchical.")
        model=factories[name]();labels=model.fit_predict(values);valid=labels!=-1
        count=len(set(labels)-{-1});metrics={"groups":count,"noise_rows":int((~valid).sum()),"coverage":float(valid.mean())}
        if 1<count<int(valid.sum()):
            positions=np.flatnonzero(valid)
            if len(positions)>2000:
                representatives=np.array([np.flatnonzero(labels==label)[0] for label in sorted(set(labels)-{-1})])
                remaining=np.setdiff1d(positions,representatives)
                positions=np.r_[representatives,np.random.default_rng(42).choice(remaining,2000-len(representatives),replace=False)]
            metrics["silhouette"]=float(silhouette_score(values[positions],labels[positions]))
            metrics["davies_bouldin"]=float(davies_bouldin_score(values[valid],labels[valid]))
        if hasattr(model,"inertia_"):metrics["inertia"]=float(model.inertia_)
        return labels,metrics
    labels,metrics=run(algorithm)
    projection=PCA(n_components=min(2,values.shape[1])).fit_transform(values)
    if projection.shape[1]==1:projection=np.column_stack([projection[:,0],np.zeros(len(frame))])
    indices=np.random.default_rng(42).choice(len(frame),min(len(frame),2000),replace=False)
    profiles=[]
    for label in sorted(set(labels)):
        rows=numeric.iloc[np.flatnonzero(labels==label)]
        profiles.append({"label":int(label),"rows":len(rows),"means":{c:(float(rows[c].mean()) if rows[c].notna().any() else None) for c in columns}})
    comparison=[]
    if compare:
        for name in ["K-Means","DBSCAN","Hierarchical"]:
            _,m=run(name);comparison.append({"algorithm":name,**m})
    scan=[]
    if algorithm=="K-Means":
        for k in range(2,min(8,len(frame)-1,np.unique(values,axis=0).shape[0])+1):
            _,m=run("K-Means",k);scan.append({"k":k,**m})
    return {"algorithm":algorithm,"columns":columns,"parameters":{"clusters":clusters,"eps":eps,"min_samples":min_samples,"scale":bool(scale)},"metrics":metrics,"labels":labels.tolist(),"profiles":profiles,"points":[[float(projection[i,0]),float(projection[i,1]),int(labels[i]),int(i)] for i in indices],"sampled":len(indices)<len(frame),"scan":scan,"comparison":comparison,"rows":len(frame),"protocol":"Exploratory whole-dataset clustering. Median imputation; optional standard scaling. PCA is for the 2D display only. Silhouette uses up to 2,000 deterministic sampled rows, retaining every group; Davies-Bouldin uses all grouped rows. Noise is excluded from both; compare coverage too. These are internal structure scores, not prediction accuracy or proof of real-world groups."}
