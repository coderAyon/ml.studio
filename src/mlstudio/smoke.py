"""Reproducible packaged-runtime acceptance check using explicitly synthetic data."""
import json
from pathlib import Path
import pandas as pd
from PySide6.QtGui import QFontDatabase
from .analyzer import analyze
from .data import DatasetVersions, load_csv
from .report import export_report
from .training import train, predict


def run(window, application, directory: str):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The offscreen Qt plugin does not discover Windows fonts automatically.
    if not QFontDatabase.families() and Path("C:/Windows/Fonts/segoeui.ttf").exists():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    source = directory / "synthetic-smoke-data.csv"
    pd.DataFrame({"feature": range(80), "category": ["a", "b"]*40, "target": [0, 1]*40}).to_csv(source, index=False)
    frame = load_csv(str(source))
    issues = analyze(frame)
    result = train(frame, "target", "Logistic Regression")
    assert len(predict(result, frame.drop(columns="target"))) == 80
    window.source = str(source)
    window.dataset = DatasetVersions(frame)
    window.issues = issues
    window.refresh_data()
    window.navigation.setCurrentRow(4)
    window.chart_x.setCurrentText("feature")
    window.draw_chart()
    window.figure.savefig(directory / "synthetic-chart.png")
    window.show()
    application.processEvents()
    window.grab().save(str(directory / "window.png"))
    export_report(directory / "synthetic-report.html", frame, issues, [], {"metrics": result.metrics, "metadata": result.metadata})
    (directory / "result.json").write_text(json.dumps({"status": "passed", "data": "synthetic acceptance fixture", "checks": ["Qt startup", "CSV import", "analysis", "training", "prediction", "chart rendering", "HTML export"], "metrics": result.metrics}, indent=2), encoding="utf-8")
    window.close()
