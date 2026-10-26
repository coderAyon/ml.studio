"""Native WebEngine integration checks with labeled synthetic fixtures."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import QEventLoop, QTimer
from .analyzer import analyze
from .data import DatasetVersions


def evaluate(window, script):
    result = []
    loop = QEventLoop()
    window.web_page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    if not result:
        raise TimeoutError("Embedded UI JavaScript did not respond.")
    return result[0]


def wait_for(window, expression, timeout=30):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if evaluate(window, expression):
            return
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
    raise TimeoutError(f"Embedded UI condition timed out: {expression}")


def capture(window, directory, page):
    evaluate(window, f"studioUI.navigate('{page}'); true")
    window.raise_()
    window.activateWindow()
    loop = QEventLoop()
    QTimer.singleShot(1500, loop.quit)
    loop.exec()
    window.grab().save(str(directory / f"{page}.png"))


def run(window, application, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    window.show()
    wait_for(window, "document.body.dataset.ready === 'true'")
    capture(window, directory, "home")
    checks = ["Embedded WebEngine startup", "Offline asset load", "QWebChannel handshake"]
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"measurement": np.r_[rng.normal(3, 1, 60), rng.normal(7, 1, 60)], "category": [" Control", "control", "Treatment"] * 40, "constant": [1]*120, "outcome": [0]*60+[1]*60})
    frame.loc[3:9, "measurement"] = np.nan
    source = directory / "synthetic-ui-test.csv"
    frame.to_csv(source, index=False)
    controller = window.controller
    controller.source = str(source)
    controller.import_options = {"delimiter": ",", "encoding": "utf-8-sig", "header": True, "decimal": "."}
    controller.dataset, controller.issues = DatasetVersions(frame), analyze(frame)
    controller.refresh_data()
    controller.target.setCurrentText("outcome")
    window.bridge.publish()
    wait_for(window, "window.appState?.dataset?.rows === 120")
    assert evaluate(window, "document.querySelectorAll('.automatic-chart svg').length === 0")
    evaluate(window, "studioUI.navigate('analytics');true")
    assert evaluate(window, "document.querySelectorAll('.automatic-chart svg').length === 3")
    assert evaluate(window, "document.querySelector('#content').textContent.includes('Your CSV, visualized automatically')")
    checks.append("Histogram, scatter and category bars in Visual Analytics")
    capture(window, directory, "home")
    capture(window, directory, "data")
    capture(window, directory, "analyzer")
    assert evaluate(window, "document.querySelectorAll('.plan-step').length > 0")
    checks.append("Data-specific preprocessing steps with reasons and previews")
    evaluate(window, "studioUI.command('preview',{operation:'Trim whitespace',column:'category'}); true")
    wait_for(window, "!!document.querySelector('[data-action=apply_preview]')")
    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec()
    window.grab().save(str(directory / "preview.png"))
    evaluate(window, "document.querySelector('[data-action=apply_preview]').click(); true")
    wait_for(window, "window.appState?.dataset?.version === 1 && !window.appState.busy")
    assert controller.dataset.current.category.iloc[0] == "Control"
    checks.append("Preview and Apply through the web bridge")
    evaluate(window, "studioUI.command('undo'); true")
    wait_for(window, "window.appState?.dataset?.version === 0 && !window.appState.busy")
    checks.append("Dataset undo through the web bridge")
    capture(window, directory, "workflow")
    evaluate(window, "document.querySelector('[data-action=starter]').click(); true")
    wait_for(window, "Object.keys(window.appState?.workflow?.nodes||{}).length === 3")
    assert controller.canvas.graph.order() == ["1", "2", "3"]
    capture(window, directory, "workflow")
    checks.append("Real workflow creation from the canvas toolbar")
    evaluate(window, "studioUI.navigate('models'); document.querySelector('[data-action=train]').click(); true")
    wait_for(window, "!!window.appState?.model && !window.appState.busy")
    assert controller.result is not None
    capture(window, directory, "models")
    checks.append("Training and metric rendering from the embedded UI")
    evaluate(window, "document.querySelector('[data-action=compare]').click(); true")
    wait_for(window, "window.appState?.comparison?.length === 6 && !window.appState.busy")
    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec()
    assert evaluate(window, "document.querySelector('#content').textContent.includes('Comparison complete')")
    assert evaluate(window, "document.querySelector('.comparison-explanation').textContent.includes('F1 (weighted)')")
    checks.append("Observed model leaders, metric explanations, ties and limitations")
    window.grab().save(str(directory / "comparison.png"))
    checks.append("Six-model comparison rendered with actual metrics")
    evaluate(window, "studioUI.navigate('workflow'); document.querySelector('[data-action=run_workflow]').click(); true")
    wait_for(window, "!!window.appState?.model && !window.appState.busy")
    checks.append("Workflow training executed through the web canvas")
    evaluate(window, "studioUI.command('chart',{type:'Histogram',x:'measurement',y:'outcome'}); true")
    wait_for(window, "!!window.appState?.chart")
    capture(window, directory, "analytics")
    checks.append("Actual Matplotlib chart displayed in the web interface")
    for page in ["history", "predictions", "reports", "settings", "deep"]:
        capture(window, directory, page)
    (directory / "result.json").write_text(json.dumps({"status": "passed", "fixture": "Explicitly synthetic UI acceptance data", "checks": checks, "version": "0.3.0"}, indent=2), encoding="utf-8")
    window.close()
