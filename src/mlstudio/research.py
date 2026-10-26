"""Local controlled research experiments with a reserved final test set."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .training import ALGORITHMS, REGRESSION
from .safety import safety_advice
from .error_analysis import error_report

VARIANTS={"baseline":"Saved preparation plan", "mean":"Saved plan + mean filling", "unscaled":"Saved plan without scaling", "indicator":"Saved plan + missing flags"}


@dataclass
class ResearchConfig:
    target: str
    task: str = "classification"
    protocol: str = "random"
    group: str = ""
    time_column: str = ""
    excluded: list[str] = field(default_factory=list)
    folds: int = 3
    repeats: int = 2
    seed: int = 42
    preparation: dict | None = None


def pipeline_for(x, algorithm, variant, seed,preparation=None):
    if variant not in VARIANTS:raise ValueError("Unknown preprocessing variant.")
    if algorithm not in ALGORITHMS:raise ValueError("Unknown algorithm.")
    from .training_prep import validate_preparation
    from .training import build_pipeline
    p=validate_preparation(preparation)
    if variant=='mean':p['imputation']='mean'
    if variant=='unscaled':p['scaling']='none'
    if variant=='indicator':p['missing_indicators']=True
    return build_pipeline(x,algorithm,p,seed)


def metric_values(y,pred,task):
    if task=="regression":
        result={"MAE":float(mean_absolute_error(y,pred)),"RMSE":float(np.sqrt(mean_squared_error(y,pred))),"R2":float(r2_score(y,pred)) if len(y)>1 else None}
    else:
        result={"Accuracy":float(accuracy_score(y,pred)),"Balanced accuracy":float(balanced_accuracy_score(y,pred)),"Macro F1":float(f1_score(y,pred,average="macro",zero_division=0)),"Weighted F1":float(f1_score(y,pred,average="weighted",zero_division=0))}
    return {k:v if v is None or np.isfinite(v) else None for k,v in result.items()}


class ResearchSession:
    """One dataset/config/split design. The final test may be used once per session."""
    def __init__(self,frame,config,split_manifest=None):
        self.config=config
        self.frame=frame.reset_index(drop=True).copy(deep=True)
        self._validate()
        from .training_prep import validate_preparation
        self.preparation=validate_preparation(config.preparation,frame.columns,config.target)
        excluded=set(config.excluded+self.preparation['excluded']+[config.target]+([config.group] if config.protocol=="group" else [])+([config.time_column] if config.protocol=="time" else []))
        self.features=[c for c in frame if c not in excluded]
        if not self.features:raise ValueError("No input features remain after exclusions.")
        self.x=self.frame[self.features].copy()
        width=sum(1 if pd.api.types.is_numeric_dtype(self.x[c]) else min(50,max(1,self.x[c].nunique())) for c in self.x)
        if len(self.x)*width>5_000_000:raise ValueError("Encoded research data would exceed the memory budget. Exclude high-cardinality identifiers or use a reviewed extract.")
        for c in self.x:
            if not pd.api.types.is_numeric_dtype(self.x[c]):
                self.x[c]=self.x[c].map(lambda v:str(v) if pd.notna(v) else np.nan).astype(object)
        self.y=self.frame[config.target].copy()
        if config.task=="regression":self.y=pd.to_numeric(self.y,errors="raise")
        self.development_positions,self.test_positions,self.folds=self._splits()
        if split_manifest is not None:
            # Only accept the exact deterministic design, never arbitrary indices.
            if split_manifest!=self.manifest():raise ValueError("Saved splits do not match the dataset/configuration.")
        self.final_used=False
        self.comparison=None
        self.impact_result=None
        self.final_result=None
        self.final_pipeline=None
        self.run_log=[]

    def _validate(self):
        cfg=self.config;frame=self.frame
        if not 20<=len(frame)<=20000 or len(frame.columns)>200:
            raise ValueError("Research tools support 20–20,000 rows and at most 200 input columns. Use a reviewed extract for larger data.")
        if cfg.target not in frame:raise ValueError("Select a valid target for research experiments.")
        if cfg.task not in {"classification","regression"} or cfg.protocol not in {"random","group","time"}:raise ValueError("Invalid task or validation protocol.")
        if not 2<=cfg.folds<=5 or not 1<=cfg.repeats<=3:raise ValueError("Use 2–5 folds and 1–3 repeats.")
        if any(c not in frame for c in cfg.excluded):raise ValueError("An excluded feature does not exist.")
        if frame[cfg.target].isna().any():raise ValueError("Target contains missing labels; review them before running experiments.")
        if cfg.task=="classification" and (frame[cfg.target].nunique()<2 or frame[cfg.target].value_counts().min()<cfg.folds+2):raise ValueError("Too few class examples for reserved test plus stratified CV. Collect more labels or reduce folds.")
        if cfg.task=="regression":
            numeric=pd.to_numeric(frame[cfg.target],errors="raise")
            if not np.isfinite(numeric).all() or numeric.nunique()<2:raise ValueError("Regression needs a finite, nonconstant numeric target.")
        values=frame.select_dtypes(include="number").to_numpy(dtype=float,na_value=np.nan)
        if np.isinf(values).any():raise ValueError("Infinite numeric values must be reviewed before research training.")
        if cfg.protocol=="group":
            if cfg.group not in frame or cfg.group==cfg.target:raise ValueError("Choose a group identifier different from the target.")
            if frame[cfg.group].isna().any():raise ValueError("Group identifiers contain missing values.")
            if frame[cfg.group].nunique()<cfg.folds+1:raise ValueError("Not enough independent groups for final reservation plus group folds.")
        if cfg.protocol=="time":
            if cfg.time_column not in frame or cfg.time_column==cfg.target:raise ValueError("Choose a valid time column different from the target.")
            if pd.api.types.is_numeric_dtype(frame[cfg.time_column]):raise ValueError("Use explicit date/time strings, not ambiguous numeric timestamps.")
            dates=pd.to_datetime(frame[cfg.time_column],errors="coerce",utc=True,format="mixed")
            if dates.isna().any() or dates.nunique()<cfg.folds+2:raise ValueError("Time values must parse unambiguously and contain enough distinct timestamps.")

    def _splits(self):
        cfg=self.config;positions=np.arange(len(self.frame))
        if cfg.protocol=="time":
            dates=pd.to_datetime(self.frame[cfg.time_column],utc=True,format="mixed")
            unique=np.sort(dates.unique());boundary=unique[max(1,int(len(unique)*.8))]
            development=positions[(dates<boundary).to_numpy()]
            final=positions[(dates>=boundary).to_numpy()]
            development=development[np.argsort(dates.iloc[development].to_numpy(),kind="stable")]
            blocks=np.sort(dates.iloc[development].unique())
            folds=[]
            for tr,va in TimeSeriesSplit(n_splits=cfg.folds).split(blocks):
                train=development[dates.iloc[development].isin(blocks[tr]).to_numpy()]
                valid=development[dates.iloc[development].isin(blocks[va]).to_numpy()]
                folds.append((train.tolist(),valid.tolist()))
        elif cfg.protocol=="group":
            groups=self.frame[cfg.group].astype(str)
            train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg.seed).split(self.frame,groups=groups))
            development,final=positions[train],positions[test];folds=[]
            for repeat in range(cfg.repeats):
                splitter=GroupKFold(n_splits=cfg.folds,shuffle=True,random_state=cfg.seed+repeat)
                for tr,va in splitter.split(development,groups=groups.iloc[development]):folds.append((development[tr].tolist(),development[va].tolist()))
        else:
            development,final=train_test_split(positions,test_size=.2,random_state=cfg.seed,stratify=self.y if cfg.task=="classification" else None)
            folds=[]
            if cfg.task=="classification" and self.y.iloc[development].value_counts().min()<cfg.folds:raise ValueError("Insufficient class examples in development data for this fold count.")
            for repeat in range(cfg.repeats):
                splitter=StratifiedKFold(cfg.folds,shuffle=True,random_state=cfg.seed+repeat) if cfg.task=="classification" else KFold(cfg.folds,shuffle=True,random_state=cfg.seed+repeat)
                for tr,va in splitter.split(development,self.y.iloc[development]):folds.append((development[tr].tolist(),development[va].tolist()))
        for train,valid in folds:
            if not train or not valid:raise ValueError("A validation fold has no rows.")
            if cfg.task=="classification" and self.y.iloc[train].nunique()<2:raise ValueError("A training fold has fewer than two classes. Review group/time design or obtain more data.")
        return development.tolist(),final.tolist(),folds

    @property
    def metric(self):return "Macro F1" if self.config.task=="classification" else "RMSE"

    def manifest(self):
        return {"development":self.development_positions,"final_test":self.test_positions,"folds":[{"train":a,"validation":b} for a,b in self.folds]}

    def _check_candidate(self,algorithm):
        if algorithm not in ALGORITHMS or (algorithm in REGRESSION)!=(self.config.task=="regression"):raise ValueError("Choose an algorithm matching the research task.")

    def _evaluate_cv(self,algorithm,variant,progress=None):
        self._check_candidate(algorithm)
        scores=[];metrics=[];durations=[]
        for i,(tr,va) in enumerate(self.folds):
            if progress:progress(f"{algorithm} / {VARIANTS[variant]} · fold {i+1}/{len(self.folds)}")
            pipeline=pipeline_for(self.x.iloc[tr],algorithm,variant,self.config.seed,self.preparation)
            started=time.perf_counter();pipeline.fit(self.x.iloc[tr],self.y.iloc[tr]);pred=pipeline.predict(self.x.iloc[va]);durations.append(time.perf_counter()-started)
            values=metric_values(self.y.iloc[va],pred,self.config.task);metrics.append(values);scores.append(values[self.metric])
        return {"algorithm":algorithm,"variant":variant,"label":VARIANTS[variant],"scores":scores,"mean":float(np.mean(scores)),"std":float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,"seconds":float(sum(durations)),"fold_metrics":metrics,"fold_ids":list(range(len(self.folds))),"parameters":ALGORITHMS[algorithm]().get_params()}

    def _baseline(self):
        scores=[]
        for tr,va in self.folds:
            model=DummyClassifier(strategy="most_frequent") if self.config.task=="classification" else DummyRegressor(strategy="mean")
            model.fit(np.zeros((len(tr),1)),self.y.iloc[tr]);pred=model.predict(np.zeros((len(va),1)))
            scores.append(metric_values(self.y.iloc[va],pred,self.config.task)[self.metric])
        return {"name":"Majority-class baseline" if self.config.task=="classification" else "Training-mean baseline","scores":scores,"mean":float(np.mean(scores)),"std":float(np.std(scores,ddof=1))}

    def compare(self,algorithms,progress=None):
        if self.final_used:raise ValueError("Final test was already inspected. Start a new independent study; do not select models against that test.")
        if not algorithms or len(set(algorithms))!=len(algorithms):raise ValueError("Choose at least one distinct algorithm.")
        models=[self._evaluate_cv(a,"baseline",progress) for a in algorithms]
        baseline=self._baseline();higher=self.config.task=="classification"
        models.sort(key=lambda r:r["mean"],reverse=higher)
        reference=np.array(models[0]["scores"])
        for row in models:
            delta=np.array(row["scores"])-reference
            row["paired_difference_to_leader"]=delta.tolist();row["mean_difference_to_leader"]=float(delta.mean())
            row["beats_baseline"]=row["mean"]>baseline["mean"] if higher else row["mean"]<baseline["mean"]
        self.comparison={"models":models,"metric":self.metric,"higher_is_better":higher,"baseline":baseline,"fold_count":len(self.folds),"development_rows":len(self.development_positions),"reserved_final_rows":len(self.test_positions),"caution":"Mean ± sample SD describes observed fold variation, not a confidence interval. Repeated folds share data and are not independent. Close scores and differences are exploratory; no statistical significance or universal winner is claimed.","protocol":self.config.protocol}
        self.run_log.append({"kind":"comparison","algorithms":list(algorithms)})
        return self.comparison

    def impact(self,algorithm,progress=None):
        if self.final_used:raise ValueError("Final test was already inspected; do not tune preprocessing against it.")
        variants=[self._evaluate_cv(algorithm,v,progress) for v in VARIANTS]
        reference=np.array(variants[0]["scores"])
        for row in variants:
            differences=np.array(row["scores"])-reference
            row["paired_differences"]=differences.tolist();row["mean_difference"]=float(differences.mean())
        self.impact_result={"algorithm":algorithm,"metric":self.metric,"higher_is_better":self.config.task=="classification","variants":variants,"caution":"Every variant uses identical development folds. Differences are observed associations under these settings, not proof of a general causal benefit. Categorical encoding is held fixed; only numeric imputation/scaling/indicators change. No numeric features or no missing values can make variants identical."}
        self.run_log.append({"kind":"impact","algorithm":algorithm})
        return self.impact_result

    def evaluate_final(self,algorithm,variant):
        if self.final_used:raise ValueError("The reserved final test was already used in this session.")
        self._check_candidate(algorithm)
        self.final_used=True  # lock on first attempt, including failed fitting
        pipeline=pipeline_for(self.x.iloc[self.development_positions],algorithm,variant,self.config.seed,self.preparation)
        started=time.perf_counter();pipeline.fit(self.x.iloc[self.development_positions],self.y.iloc[self.development_positions]);pred=pipeline.predict(self.x.iloc[self.test_positions])
        actual=self.y.iloc[self.test_positions]
        result={"algorithm":algorithm,"variant":variant,"metrics":metric_values(actual,pred,self.config.task),"seconds":time.perf_counter()-started,"predictions":[v.item() if isinstance(v,np.generic) else v for v in pred],"positions":self.test_positions,"errors":error_report(self.frame[self.features],self.test_positions,actual,pred,self.config.task),"caution":"Final test inspected once. Subsequent decisions informed by these outcomes require a new external final dataset. Restarting a session does not make the same data independent."}
        if self.config.task=="classification":
            labels=np.unique(np.concatenate([pipeline.named_steps["model"].classes_,np.asarray(actual)]))
            result["class_labels"]=[str(v) for v in labels]
            result["confusion_matrix"]=confusion_matrix(actual,pred,labels=labels).tolist()
            result["per_class"]=classification_report(actual,pred,output_dict=True,zero_division=0)
        self.final_result=result;self.final_pipeline=pipeline
        self.run_log.append({"kind":"final","algorithm":algorithm,"variant":variant})
        return result

    def snapshot(self):
        return {"config":asdict(self.config),"features":self.features,"metric":self.metric,"development_rows":len(self.development_positions),"reserved_final_rows":len(self.test_positions),"fold_count":len(self.folds),"final_used":self.final_used,"comparison":self.comparison,"impact":self.impact_result,"final":self.final_result,"safety":safety_advice(self.frame,self.config),"fold_summary":[{"fold":i+1,"training_rows":len(tr),"validation_rows":len(va)} for i,(tr,va) in enumerate(self.folds)],"time_note":"Time validation uses one forward-chaining pass; repeats are ignored." if self.config.protocol=="time" else ""}
