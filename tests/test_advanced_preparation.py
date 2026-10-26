import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from mlstudio.training import train,predict,ALGORITHMS,REGRESSION


def frame():
    f=pd.DataFrame({'x':np.arange(80,dtype=float),'z':np.sin(np.arange(80)),
                    'constant':np.ones(80),'kind':['a','b',None,'a']*20,'target':[0,1]*40})
    f.loc[[2,5,9,30],'x']=np.nan
    return f


@pytest.mark.parametrize('imputation',['median','mean','most_frequent','constant','knn'])
def test_fill_strategies_missing_categories_and_prediction(imputation):
    f=frame();original=f.copy(deep=True)
    result=train(f,'target','Decision Tree',preparation={'imputation':imputation,'constant_value':-7,
        'categorical_imputation':'constant','categorical_constant':'Unknown','missing_indicators':True})
    values=result.pipeline[:-1].transform(f.drop(columns='target'))
    assert np.isfinite(values).all()
    assert result.metadata['fitted_preparation']['output_features']==values.shape[1]
    assert any('missingindicator' in name for name in result.metadata['fitted_preparation']['encoded_columns'])
    assert any('Unknown' in name for name in result.metadata['fitted_preparation']['encoded_columns'])
    assert len(predict(result,f.drop(columns='target').head()))==5
    pd.testing.assert_frame_equal(f,original)
    imputer=result.pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer']
    if imputation=='constant':assert imputer.statistics_[0]==-7
    if imputation=='knn':assert len(imputer._fit_X)==60


@pytest.mark.parametrize('scaling',['standard','robust','minmax','maxabs','l1','l2','none'])
def test_scalers_fit_train_and_transform_future(scaling):
    f=frame();result=train(f,'target','Decision Tree',preparation={'scaling':scaling})
    numeric=result.pipeline.named_steps['preprocess'].named_transformers_['numeric']
    future=np.array([[10000.,.5,1.]])
    values=numeric.transform(future)
    assert np.isfinite(values).all()
    if scaling=='minmax':
        xt,_,_,_=train_test_split(f.drop(columns='target'),f.target,random_state=42,test_size=.25,stratify=f.target)
        assert numeric.named_steps['scaler'].data_max_[0]==xt.x.max()
        assert values[0,0]>1  # Do not silently clip unseen values.
    if scaling in {'l1','l2'}:
        np.testing.assert_allclose(np.linalg.norm(values,ord=1 if scaling=='l1' else 2,axis=1),1)


@pytest.mark.parametrize('reduction',['f_test','pca'])
@pytest.mark.parametrize('bins',['uniform','quantile'])
def test_binning_and_reduction_are_fitted_and_predictable(reduction,bins):
    f=frame();result=train(f,'target','Logistic Regression',preparation={'discretization':bins,
        'bins':3,'remove_constant':True,'feature_selection':reduction,'selection_k':2,'pca_components':2})
    assert result.metadata['fitted_preparation']['output_features']==2
    assert len(predict(result,f.drop(columns='target').head()))==5
    pre=result.pipeline.named_steps['preprocess'].named_transformers_['numeric']
    xt,_,_,_=train_test_split(f.drop(columns='target'),f.target,random_state=42,test_size=.25,stratify=f.target)
    assert pre.named_steps['discretizer'].bin_edges_[0][-1]==xt.x.max()


def test_knn_budget_and_pca_errors_are_actionable():
    with pytest.raises(ValueError,match='2,000'):train(pd.concat([frame()]*26,ignore_index=True),'target','Decision Tree',preparation={'imputation':'knn'})
    with pytest.raises(ValueError,match='PCA'):train(frame(),'target','Decision Tree',preparation={'feature_selection':'pca','pca_components':50})


@pytest.mark.parametrize('algorithm',list(ALGORITHMS))
def test_every_registered_algorithm_can_train_and_predict(algorithm):
    f=frame()
    if algorithm in REGRESSION:f['target']=np.arange(80)*.3+np.cos(np.arange(80))
    result=train(f,'target',algorithm)
    assert len(predict(result,f.drop(columns='target').head()))==5
    assert result.metadata['training_rows']==60 and result.metadata['test_rows']==20
