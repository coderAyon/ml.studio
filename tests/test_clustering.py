import numpy as np
import pandas as pd
import pytest
from mlstudio.clustering import study


def fixture():
    rng=np.random.default_rng(7)
    return pd.DataFrame({"x":np.r_[rng.normal(-4,.2,35),rng.normal(4,.2,35)],"y":np.r_[rng.normal(-3,.2,35),rng.normal(3,.2,35)],"outcome":[0]*35+[1]*35})


def test_clustering_comparison_real_labels_profiles_and_same_features():
    frame=fixture();frame.loc[3,"x"]=np.nan
    before=frame.copy(deep=True)
    r=study(frame,["x","y"],clusters=2,compare=True)
    assert len(r["labels"])==len(frame)
    assert r["columns"]==["x","y"]
    assert r["metrics"]["groups"]==2 and r["metrics"]["silhouette"]>.8
    assert sum(p["rows"] for p in r["profiles"])==len(frame)
    for p in r["profiles"]:
        actual=frame.loc[np.array(r["labels"])==p["label"],"x"].mean()
        assert p["means"]["x"]==pytest.approx(actual)
    assert {m["algorithm"] for m in r["comparison"]}=={"K-Means","DBSCAN","Hierarchical"}
    assert r["scan"][0]["k"]==2
    pd.testing.assert_frame_equal(frame,before)


def test_dbscan_all_noise_and_single_feature_projection():
    r=study(fixture(),["x"],algorithm="DBSCAN",eps=.000001,min_samples=4)
    assert r["metrics"]["noise_rows"]==70
    assert r["metrics"]["coverage"]==0
    assert "silhouette" not in r["metrics"]
    assert all(p[1]==0 for p in r["points"])
    assert r["profiles"][0]["label"]==-1


@pytest.mark.parametrize("columns,kwargs",[([],{}),(["unknown"],{}),(["x","x"],{}),(["x"],{"clusters":200}),(["x"],{"clusters":2.5}),(["x"],{"eps":float('inf')}),(["x"],{"min_samples":1})])
def test_invalid_inputs_fail_clearly(columns,kwargs):
    with pytest.raises(ValueError):study(fixture(),columns,**kwargs)


def test_no_finite_or_variable_measurements_rejected():
    with pytest.raises(ValueError):study(pd.DataFrame({"x":[1.,2.,np.inf]}),["x"])
    with pytest.raises(ValueError):study(pd.DataFrame({"x":[None]*5}),["x"])
    with pytest.raises(ValueError):study(pd.DataFrame({"x":[1]*5}),["x"])
