"""Contracts at the local HTML/Python boundary, including network isolation."""
import json
import pandas as pd
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from mlstudio.app import MainWindow
from mlstudio.desktop import ASSETS, StudioBridge, allowed_url
from mlstudio.data import DatasetVersions
from mlstudio.analyzer import analyze


def controller_with_data(tmp_path):
    app = QApplication.instance() or QApplication([])
    controller = MainWindow(tmp_path)
    frame = pd.DataFrame({"x": [1, 2, 3], "name": [" A", "b", "c"]})
    controller.source = str(tmp_path / "data.csv")
    controller.dataset = DatasetVersions(frame)
    controller.issues = analyze(frame)
    controller.refresh_data()
    return app, controller, StudioBridge(controller)


def test_bridge_snapshot_preview_and_allowlist(tmp_path):
    app, controller, bridge = controller_with_data(tmp_path)
    snapshot = bridge.snapshot()
    assert snapshot["dataset"]["rows"] == 3
    assert snapshot["dataset"]["preview"]["data"][0][1] == " A"
    bridge.command("preview", json.dumps({"operation": "Trim whitespace", "column": "name"}))
    assert bridge.preview["after"]["data"][0][1] == "A"
    assert controller.dataset.current.name.iloc[0] == " A"
    bridge.command("cancel_preview", "{}")
    assert bridge.preview is None
    errors = []
    bridge.notice.connect(lambda text: errors.append(json.loads(text)))
    bridge.command("execute_python", '{"code":"arbitrary"}')
    assert errors[-1]["kind"] == "error"
    assert controller.dataset.current.name.iloc[0] == " A"
    controller.close()


def test_empty_numeric_dashboard_and_offline_urls(tmp_path):
    app, controller, bridge = controller_with_data(tmp_path)
    controller.dataset = DatasetVersions(pd.DataFrame({"empty": [float('nan')]*3}))
    assert bridge.snapshot()["dataset"]["distribution"] is None
    assert allowed_url(QUrl.fromLocalFile(str(ASSETS / "index.html")))
    assert allowed_url(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
    assert not allowed_url(QUrl("https://example.com"))
    assert not allowed_url(QUrl("http://127.0.0.1:8000"))
    assert not allowed_url(QUrl.fromLocalFile(str(tmp_path / "private.csv")))
    controller.close()


def test_research_invalid_design_does_not_corrupt_saved_settings(tmp_path):
    app, controller, bridge = controller_with_data(tmp_path)
    original=dict(bridge.research.options)
    bridge.command('research_config',json.dumps({'folds':99,'protocol':'invalid'}))
    assert bridge.research.options==original
    controller.close()


def test_remove_dataset_preserves_files_and_clears_session(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    app, controller, bridge = controller_with_data(tmp_path)
    source = tmp_path / "data.csv"
    controller.dataset.current.to_csv(source, index=False)
    original = source.read_bytes()
    project = tmp_path / "saved.json"
    project.write_text("saved project", encoding="utf-8")
    controller.active_project_path = str(project)
    bridge.preview = {"unsaved": True}
    bridge.chart = "old chart"
    bridge.table_options = {"search": "old"}
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.No)
    bridge.command("remove_dataset")
    assert controller.dataset is not None
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    controller.busy = True
    bridge.command("remove_dataset")
    assert controller.dataset is not None
    controller.busy = False
    bridge.command("remove_dataset")
    snapshot = bridge.snapshot()
    assert snapshot["dataset"] is None and snapshot["model"] is None
    assert snapshot["issues"] == [] and snapshot["preview"] is None
    assert bridge.chart == "" and bridge.table_options == {}
    assert controller.active_project_path is None
    assert controller.target.count() == 0
    assert source.read_bytes() == original
    assert project.read_text(encoding="utf-8") == "saved project"
    controller.close()
