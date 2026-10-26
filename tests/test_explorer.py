import json
import pandas as pd
import pytest
from mlstudio.analyzer import profile
from mlstudio.data import DatasetVersions, Transformation
from test_bridge import controller_with_data


def test_all_row_search_pagination_and_stable_sort(tmp_path):
    app, controller, bridge = controller_with_data(tmp_path)
    frame = pd.DataFrame({"x":range(245),"label":["keep"]*244+["last unique"]})
    controller.dataset = DatasetVersions(frame)
    bridge.command("table_query",json.dumps({"search":"last unique","page":0}))
    data=bridge.snapshot()["dataset"]
    assert data["filtered_rows"] == 1
    assert data["preview"]["index"] == [244]
    bridge.command("table_query",json.dumps({"sort":"x","descending":True,"page":1}))
    data=bridge.snapshot()["dataset"]
    assert data["preview"]["data"][0][0] == 144
    assert len(data["preview"]["data"]) == 100
    controller.close()


def test_profile_reports_actual_statistics_and_nullable_counts():
    f = profile(pd.DataFrame({"x":[1.,2.,3.,None],"label":["a","a","b",None]}))["fields"]
    assert f[0]["mean"]==f[0]["median"]==2
    assert f[0]["std"]==1
    assert f[0]["missing_percent"]==25
    assert f[1]["top_values"][0]=={"value":"a","count":2,"percent":50.0}


def test_renaming_filtering_type_correction_and_undo():
    frame=pd.DataFrame({"name":[" Alpha","Beta"],"number":["1","2"]})
    v=DatasetVersions(frame)
    v.apply(Transformation("Rename column",["name"],"category"))
    v.apply(Transformation("Convert to numeric",["number"]))
    v.apply(Transformation("Filter rows",["category"],"alpha"))
    assert len(v.current)==1 and v.current.number.iloc[0]==1
    v.undo()
    assert len(v.current)==2
    v.redo()
    assert len(v.current)==1
    assert list(frame)==["name","number"]
    with pytest.raises(ValueError):Transformation("Rename column",["category"],"number").preview(v.current)
    with pytest.raises(ValueError):Transformation("Rename column",["category"]," number ").preview(v.current)


def test_invalid_bulk_preview_keeps_original_unchanged(tmp_path):
    app,controller,bridge=controller_with_data(tmp_path)
    errors=[]
    bridge.notice.connect(errors.append)
    bridge.command("bulk_preview",json.dumps({"operations":[{"operation":"Trim whitespace","columns":["name"]},{"operation":"Convert to numeric","columns":["name"]}]}))
    assert errors and bridge.preview is None
    assert controller.dataset.current.name.iloc[0]==" A"
    assert controller.dataset.index==0
    controller.close()
