import pandas as pd
import pytest
from mlstudio.training import train,compare


def test_train_and_comparison_respect_same_selected_test_share():
    frame=pd.DataFrame({'x':range(100),'category':['A','B']*50,'target':[0,1]*50})
    result=train(frame,'target','Decision Tree',test_size=.2)
    assert result.metadata['training_rows']==80 and result.metadata['test_rows']==20
    assert result.metadata['test_size']==.2
    results=compare(frame,'target',['Decision Tree','Logistic Regression'],test_size=.3)
    assert all(r.metadata['training_rows']==70 and r.metadata['test_rows']==30 for r in results)
    for size in [True,0.,.8,float('nan')]:
        with pytest.raises(ValueError,match='test share'):train(frame,'target','Decision Tree',test_size=size)
