"""Read-only audit probes, not passing regression tests or benchmark claims.

Runs on self-generated data in a temporary directory. Does not alter application
code, user datasets, settings or history. Some observations deliberately expose
incorrect behavior in baseline 3f24d03; a successful script is not product approval.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlstudio.analyzer import analyze
from mlstudio.comparison_tools import ComparisonTools
from mlstudio.data import DatasetVersions, Transformation, load_csv
from mlstudio.evaluation import explain_inputs
from mlstudio.flow_engine import FlowEngine
from mlstudio.model_io import load_model, save_model
from mlstudio.preprocessing import apply_operation, inspect_finding, preview_change
from mlstudio.project_data import pack_frame, unpack_frame
from mlstudio.research import ResearchConfig, ResearchSession
from mlstudio.research_export import export_pack, replay_pack
from mlstudio.training import train
from mlstudio.workflow import Graph


def run():
    observations = {}
    with tempfile.TemporaryDirectory(prefix="mlstudio-readonly-audit-") as directory:
        temp = Path(directory)
        source = temp / "large-identifiers.csv"
        source.write_text("identifier\n0\n18446744073709551615\n", encoding="utf-8")
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        before = load_csv(source)
        after = unpack_frame(pack_frame(before))
        observations["unsigned_snapshot"] = {
            "before": [str(x) for x in before.identifier],
            "after": [str(x) for x in after.identifier],
            "before_dtype": str(before.identifier.dtype),
            "after_dtype": str(after.identifier.dtype),
            "source_preserved": hashlib.sha256(source.read_bytes()).hexdigest() == original_hash,
        }

        counts = []
        for values, title in [(["1", "2", "3", "4", "bad"], "Possibly mixed numeric values"),
                              (["A", "a", "b", "c"], "Equivalent category labels")]:
            frame = pd.DataFrame({"value": values})
            issue = next(i for i in analyze(frame) if i.title == title)
            counts.append({"title": title, "summary": issue.count,
                           "detail": inspect_finding(frame, issue)["affected_count"]})
        observations["finding_counts"] = counts

        frame = pd.DataFrame({"label": pd.Categorical([" A", "b", "c"])})
        categorical = {"applied_values": apply_operation(frame, "Trim whitespace", "label").label.tolist()}
        try:
            preview_change(frame, "Trim whitespace", "label")
            categorical["preview_error"] = None
        except (ValueError, TypeError) as exc:
            categorical["preview_error"] = str(exc)
        observations["categorical_preview"] = categorical

        versions = DatasetVersions(pd.DataFrame({"x": range(80), "category": ["A", "B"] * 40,
                                                "label": [0, 1] * 40}))
        graph = {"nodes": {"1": {"type": "Dataset", "position": [0, 0], "params": {"source": "workspace"}},
                           "2": {"type": "Data table", "position": [200, 0]}}, "edges": [["1", "2"]]}
        engine = FlowEngine()

        def execute():
            return engine.run(graph, versions.current, "label", "Decision Tree", None,
                              {"workspace": (id(versions), versions.index), "files": {}}, strict_sources=True)

        versions.apply(Transformation("Filter rows", ["category"], "A"))
        execute()
        bridge = SimpleNamespace(controller=SimpleNamespace(dataset=versions, result=None))
        tools = ComparisonTools(bridge)
        tools.rules, tools.rules_key = {"audit_marker": "rules-for-A"}, tools.key()
        versions.undo()
        hidden_on_undo = tools.state()["association"]["result"] is None
        versions.apply(Transformation("Filter rows", ["category"], "B"))
        result = execute()
        observations["revision_cache_alias"] = {
            "version": versions.index, "actual_first_x": int(versions.current.x.iloc[0]),
            "cached_first_x": int(engine.values["2"].frame.x.iloc[0]),
            "computed": result["computed"], "cached": result["cached"],
            "association_hidden_on_undo": hidden_on_undo,
            "old_association_revived": tools.state()["association"]["result"] is not None,
        }

        frame = pd.DataFrame({"label": ["A", "A", "B"], "x": [1, 1, 2]})
        legacy = Graph(graph["nodes"], graph["edges"]).execute(frame, "label", "Decision Tree")["2"]
        reactive = FlowEngine()
        reactive.run(graph, frame, "label", "Decision Tree", None, "audit", strict_sources=True)
        observations["legacy_execution_divergence"] = {
            "input_rows": len(frame), "legacy_data_table_rows": len(legacy),
            "reactive_data_table_rows": len(reactive.values["2"].frame),
        }

        frame = pd.DataFrame({"x": np.arange(80, dtype=float), "z": np.sin(np.arange(80)), "label": [0, 1] * 40})
        model = train(frame, "label", "Logistic Regression")
        renamed = frame.rename(columns={"x": "z", "z": "x"})
        explanation = explain_inputs(model, renamed)
        observations["explanation_schema_guard"] = {
            "renamed_columns": list(renamed), "accepted": bool(explanation["rows"]),
            "recorded_columns": model.metadata["dataset_columns"],
        }

        # Only our own fitted skops pipeline is loaded; no pickle/joblib input.
        safe = temp / "own-model.mlmodel"
        save_model(safe, model)
        with zipfile.ZipFile(safe) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            pipeline = archive.read("pipeline.skops")
        manifest["versions"], manifest["metadata"] = {}, {}
        altered = temp / "own-model-incomplete-manifest.mlmodel"
        with zipfile.ZipFile(altered, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("pipeline.skops", pipeline)
        loaded = load_model(altered)
        observations["incomplete_model_manifest"] = {
            "accepted_empty_versions_and_metadata": loaded.metadata == {},
            "security_scope": "Self-generated safe pipeline; no code-execution exploit tested or established.",
        }

        frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=100, freq="us"),
                              "x": np.arange(100, dtype=float), "label": [0, 1] * 50})
        session = ResearchSession(frame, ResearchConfig(target="label", protocol="time", time_column="time", folds=2, repeats=1))
        pack = temp / "own-research.zip"
        export_pack(pack, session, {"fixture": "synthetic audit"})
        with zipfile.ZipFile(pack) as archive:
            restored = pd.read_json(io.StringIO(archive.read("frame.json").decode()), orient="table")
            # Exporter creates these fixed names from our own app source.
            archive.extractall(temp / "own-replay")
        replay_error = None
        try:
            replay_pack(temp / "own-replay")
        except ValueError as exc:
            replay_error = str(exc)
        observations["research_time_precision"] = {
            "distinct_before": int(frame.time.nunique()), "distinct_after": int(restored.time.nunique()),
            "replay_error": replay_error,
        }

    return {"baseline": "3f24d03", "purpose": "Audit observations; known bugs remain unfixed.",
            "observations": observations}


if __name__ == "__main__":
    report = run()
    output = Path(__file__).with_name("findings.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
