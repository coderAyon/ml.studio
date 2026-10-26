import json
import socket
import numpy as np
import pandas as pd
import pytest
from mlstudio.data import DatasetVersions, Transformation, load_csv
from mlstudio.analyzer import analyze, profile
from mlstudio.training import train, compare, cluster, predict
from mlstudio.workflow import Graph
from mlstudio.storage import Store
from mlstudio.report import export_report


def test_csv_formats_and_errors(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text('name;value\n"a;b";2\nx;3\n', encoding="utf-8-sig")
    frame = load_csv(str(path))
    assert frame.shape == (2, 2)
    assert frame.iloc[0, 0] == "a;b"
    assert pd.api.types.is_numeric_dtype(frame.value)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_csv(str(path))
    path.write_text("a,b\n1,2\n3,4,5\n")
    with pytest.raises(pd.errors.ParserError):
        load_csv(str(path), delimiter=",")


def test_unicode_sample_boundary_and_duplicate_names(tmp_path):
    path = tmp_path / "unicode.csv"
    path.write_text("label,label\n" + "é"*65530 + ",value\n", encoding="utf-8")
    frame = load_csv(str(path), delimiter=",")
    assert frame.attrs["duplicate_headers"] == ["label"]
    assert any(i.title == "Duplicate source column names" for i in analyze(frame))


def test_findings():
    frame = pd.DataFrame({"x": [1, 1, 2, 3, 4, 100, None], "constant": [4]*7, "label": [" A", " A", "a", "b", "c", "d", None]})
    issues = analyze(frame)
    titles = {i.title for i in issues}
    assert {"Missing values", "Repeated rows", "Empty or constant column", "Potential outliers", "Surrounding whitespace", "Equivalent category labels"} <= titles
    assert all(i.recommendation and i.risk and i.method for i in issues)
    assert next(i for i in issues if i.title == "Potential outliers").status == "Potential Issue"
    assert profile(frame)["rows"] == 7


def test_versions_preserve_original_and_branch():
    original = pd.DataFrame({"x": [1., None, 1.]})
    versions = DatasetVersions(original)
    transform = Transformation("Fill missing", ["x"])
    assert transform.preview(original).x.isna().sum() == 0
    assert original.x.isna().sum() == 1
    versions.apply(transform)
    versions.undo()
    assert versions.current.equals(original)
    versions.redo()
    assert not versions.current.x.isna().any()
    versions.undo()
    versions.apply(Transformation("Remove duplicates", []))
    assert len(versions.current) == 2
    assert len(versions.log) == 1


def test_graph_validation_execution_and_roundtrip(tmp_path):
    graph = Graph({"1": {"type": "Dataset"}, "2": {"type": "Remove duplicates"}}, [("1", "2")])
    assert graph.order() == ["1", "2"]
    output = graph.execute(pd.DataFrame({"x": [1, 1, 2]}), "", "")
    assert len(output["2"]) == 2
    path = tmp_path / "flow.json"
    graph.save(path)
    assert Graph.load(path).order() == ["1", "2"]
    with pytest.raises(ValueError):
        Graph({"1": {"type": "Train model"}, "2": {"type": "Remove duplicates"}}, [("1", "2")]).order()
    with pytest.raises(ValueError, match="cycle"):
        Graph({"1": {"type": "Remove duplicates"}, "2": {"type": "Remove duplicates"}}, [("1", "2"), ("2", "1")]).order()
    with pytest.raises(ValueError, match="disconnected"):
        Graph({"1": {"type": "Train model"}}, []).order()


def test_training_metrics_and_train_only_statistics(monkeypatch):
    from sklearn.model_selection import train_test_split
    frame = pd.DataFrame({"x": list(range(80)), "category": ["a", "b"]*40, "target": [0, 1]*40})
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Core must remain offline"))
    result = train(frame, "target", "Logistic Regression")
    x_train, _, _, _ = train_test_split(frame.drop(columns="target"), frame.target, random_state=42, test_size=.25, stratify=frame.target)
    imputer = result.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == x_train.x.median()
    assert result.metadata["training_rows"] == 60
    assert 0 <= result.metrics["Accuracy"] <= 1
    assert np.array(result.metrics["Confusion matrix"]).sum() == 20
    again = train(frame, "target", "Logistic Regression")
    assert again.metrics == result.metrics
    regression = frame.assign(target=frame.x * 2 + 5)
    assert train(regression, "target", "Linear Regression").metrics["RMSE"] < 1e-8


def test_store_and_report(tmp_path):
    store = Store(tmp_path)
    store.add("Ridge", {"MAE": 1.2})
    assert Store(tmp_path).history()[0][1] == "Ridge"
    frame = pd.DataFrame({"<script>": [1, None]})
    report = tmp_path / "report.html"
    export_report(report, frame, analyze(frame), [], {})
    assert "&lt;script&gt;" in report.read_text(encoding="utf-8")
    assert "<script>" not in report.read_text(encoding="utf-8")


def test_comparison_and_clustering():
    frame = pd.DataFrame({"x": np.arange(80), "y": np.tile([0, 1], 40)})
    results = compare(frame, "y", ["Logistic Regression", "Decision Tree"])
    assert len(results) == 2
    assert results[0].metadata["training_rows"] == results[1].metadata["training_rows"]
    assert results[0].metadata["seed"] == results[1].metadata["seed"]
    with pytest.raises(ValueError, match="same task"):
        compare(frame, "y", ["Ridge", "Decision Tree"])
    grouped = pd.DataFrame({"a": np.r_[np.arange(10)/100, 20+np.arange(10)/100], "b": np.r_[np.arange(10)/100, 20+np.arange(10)/100]})
    result = cluster(grouped, "K-Means", 2)
    assert result["metrics"]["clusters"] == 2
    assert result["metrics"]["Silhouette"] > .9
    assert len(cluster(grouped, "PCA")["embedding"]) == len(grouped)


def test_workflow_model_and_no_leaky_imputation():
    frame = pd.DataFrame({"x": range(80), "target": [0, 1]*40})
    graph = Graph({"1": {"type": "Dataset"}, "2": {"type": "Train model"}}, [("1", "2")])
    result = graph.execute(frame, "target", "Decision Tree")
    assert result["2"].metadata["test_rows"] == 20
    graph.nodes["2"]["type"] = "Fill missing"
    with pytest.raises(ValueError, match="Exploratory imputation"):
        graph.execute(frame, "target", "Decision Tree")


def test_prediction_schema_and_categorical_conversion():
    frame = pd.DataFrame({"numeric": range(80), "boolean": [True, False]*40, "target": [0, 1]*40})
    result = train(frame, "target", "Logistic Regression")
    assert len(predict(result, frame.drop(columns="target").head(5))) == 5
    with pytest.raises(ValueError, match="Missing"):
        predict(result, frame[["numeric"]])
    with pytest.raises(ValueError):
        predict(result, pd.DataFrame({"numeric": ["invalid"], "boolean": [True]}))
