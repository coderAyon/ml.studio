"""Preparation choices fitted exclusively on training rows, including outlier fences."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

DEFAULT_PREPARATION={"imputation":"median","scaling":"standard","outliers":"keep","excluded":[],
 "constant_value":0.,"categorical_imputation":"most_frequent","categorical_constant":"Missing",
 "knn_neighbors":5,"missing_indicators":False,"discretization":"none","bins":5,
 "remove_constant":False,"feature_selection":"none","selection_k":10,"pca_components":2}
EXPLORATORY_OPERATIONS={"Fill missing","Cap outliers (IQR)","Remove outlier rows (IQR)"}
MAX_CATEGORY_LEVELS = 100


def encoded_width_bound(frame, preparation=None):
    """Conservative dense width before reduction, without learning any statistics.

    Count possible numeric missing indicators and the same one-hot cap used by
    build_pipeline. This is an element budget, not a peak-process-RAM guarantee.
    """
    p = validate_preparation(preparation)
    numeric = list(frame.select_dtypes(include='number').columns)
    width = len(numeric)
    if p['missing_indicators']:
        width += sum(bool(frame[c].isna().any()) for c in numeric)
    for c in frame:
        if c not in numeric:
            # A training-fold imputer can add one category for absent values.
            width += min(MAX_CATEGORY_LEVELS, max(1, frame[c].nunique() + int(frame[c].isna().any())))
    return width


def validate_preparation(preparation,columns=None,target=None):
    if preparation is None:preparation={}
    if not isinstance(preparation,dict) or set(preparation)-set(DEFAULT_PREPARATION):
        raise ValueError("Invalid training preparation settings.")
    p={**DEFAULT_PREPARATION,**preparation}
    if p['imputation'] not in {'median','mean','most_frequent','constant','knn'} or p['scaling'] not in {'standard','robust','minmax','maxabs','l1','l2','none'} or p['outliers'] not in {'keep','cap_iqr'}:
        raise ValueError("Choose supported imputation, scaling and outlier settings.")
    if p['categorical_imputation'] not in {'most_frequent','constant'} or not isinstance(p['categorical_constant'],str) or not p['categorical_constant'].strip() or len(p['categorical_constant'])>100:
        raise ValueError('Choose mode filling or a nonempty category label up to 100 characters.')
    if isinstance(p['constant_value'],bool) or not isinstance(p['constant_value'],(int,float)) or not np.isfinite(p['constant_value']):
        raise ValueError('Numeric fill constant must be a finite number.')
    if any(type(p[key]) is not bool for key in ['missing_indicators','remove_constant']):raise ValueError('Indicator/constant-feature options must be on or off.')
    for key,lower,upper in [('knn_neighbors',1,15),('bins',2,20),('selection_k',1,100),('pca_components',1,50)]:
        if type(p[key]) is not int or not lower<=p[key]<=upper:raise ValueError(f'{key} must be an integer from {lower} to {upper}.')
    if p['discretization'] not in {'none','uniform','quantile'} or p['feature_selection'] not in {'none','f_test','pca'}:
        raise ValueError('Choose supported discretization and feature-reduction methods.')
    if not isinstance(p['excluded'],list) or any(not isinstance(c,str) for c in p['excluded']) or len(p['excluded'])!=len(set(p['excluded'])):
        raise ValueError("Choose distinct feature columns to exclude.")
    if columns is not None and any(c not in columns for c in p['excluded']):raise ValueError("An excluded feature is no longer present. Review preprocessing.")
    if target and target in p['excluded']:raise ValueError("The outcome cannot be excluded. Review the feature exclusions in Preprocessing.")
    return {**p,'excluded':list(p['excluded'])}


def numeric_pipeline(preparation):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer,KNNImputer
    from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler,Normalizer,KBinsDiscretizer
    p=validate_preparation(preparation);steps=[]
    if p['outliers']=='cap_iqr':steps.append(('outliers',FoldIQRCapper()))
    if p['imputation']=='knn':imputer=KNNImputer(n_neighbors=p['knn_neighbors'],add_indicator=p['missing_indicators'],keep_empty_features=True)
    else:imputer=SimpleImputer(strategy=p['imputation'],fill_value=p['constant_value'] if p['imputation']=='constant' else None,
                               add_indicator=p['missing_indicators'],keep_empty_features=True)
    steps.append(('imputer',imputer))
    if p['discretization']!='none':steps.append(('discretizer',KBinsDiscretizer(n_bins=p['bins'],encode='ordinal',strategy=p['discretization'],subsample=None,quantile_method='averaged_inverted_cdf')))
    scalers={'standard':StandardScaler(),'robust':RobustScaler(),'minmax':MinMaxScaler(),
             'maxabs':MaxAbsScaler(),'l1':Normalizer(norm='l1'),'l2':Normalizer(norm='l2'),'none':'passthrough'}
    steps.append(('scaler',scalers[p['scaling']]))
    return Pipeline(steps)


class BoundedFeatureSelector(BaseEstimator,TransformerMixin):
    """Fit a supervised F-test only on training data; bound k by available encoded inputs."""
    def __init__(self,k=10,regression=False):self.k=k;self.regression=regression
    def fit(self,x,y):
        from sklearn.feature_selection import SelectKBest,f_classif,f_regression
        self.n_features_in_=x.shape[1]
        self.selector_=SelectKBest(f_regression if self.regression else f_classif,k=min(self.k,x.shape[1])).fit(x,y)
        return self
    def transform(self,x):return self.selector_.transform(x)
    def get_feature_names_out(self,input_features=None):
        features=np.asarray(input_features if input_features is not None else ['x'+str(i) for i in range(self.n_features_in_)],dtype=object)
        return features[self.selector_.get_support()]


def fitted_preparation_summary(pipeline):
    """Actual fitted fields and dimensions, not advice pretending to be model evidence."""
    pre=pipeline.named_steps['preprocess'];numeric=pre.named_transformers_['numeric'];categorical=pre.named_transformers_['categorical']
    fields=[]
    if hasattr(numeric,'named_steps'):
        imputer=numeric.named_steps['imputer']
        statistics=getattr(imputer,'statistics_',None)
        if statistics is not None:
            for col,value in zip(pre.transformers_[0][2],statistics):
                fields.append({'column':col,'fill_value':float(value) if np.isfinite(value) else None})
    names=pipeline[:-1].get_feature_names_out().tolist()
    return {'encoded_columns':names,'output_features':len(names),'numeric_fill_values':fields,
            'numeric_features':list(pre.transformers_[0][2]),'categorical_features':list(pre.transformers_[1][2]),
            'steps':[name for name,_ in pipeline.steps[:-1]],
            'note':'Learned from training rows only. F-test ranking is an association score, not causal importance. PCA components replace original feature axes.'}


class FoldIQRCapper(BaseEstimator,TransformerMixin):
    """Clip numeric features using finite quartiles fitted on the current training split."""
    def fit(self,x,y=None):
        values=np.asarray(x,dtype=float)
        self.n_features_in_=values.shape[1]
        self.lower_=np.full(self.n_features_in_,-np.inf)
        self.upper_=np.full(self.n_features_in_,np.inf)
        for j in range(self.n_features_in_):
            s=values[:,j];s=s[np.isfinite(s)]
            if s.size:
                q1,q3=np.quantile(s,[.25,.75]);iqr=q3-q1
                if iqr>0 and np.isfinite(iqr):
                    self.lower_[j]=q1-1.5*iqr;self.upper_[j]=q3+1.5*iqr
        return self

    def transform(self,x):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self,'lower_')
        values=np.asarray(x,dtype=float)
        if values.shape[1]!=self.n_features_in_:raise ValueError("Numeric feature count changed.")
        return np.clip(values,self.lower_,self.upper_)

    def get_feature_names_out(self,input_features=None):
        return np.asarray(input_features if input_features is not None else ['x'+str(i) for i in range(self.n_features_in_)],dtype=object)


def assert_evaluation_ready(dataset):
    operations={op['operation'] for op in dataset.log[:dataset.index]}
    if operations & EXPLORATORY_OPERATIONS:
        raise ValueError("Undo whole-dataset imputation/outlier transformations before evaluation. In Preprocessing, choose training-only preparation so statistics are fitted inside the training split.")
