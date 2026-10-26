import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from mlstudio.app import MainWindow, THEME
import pandas as pd
from mlstudio.data import DatasetVersions, Transformation
from mlstudio.analyzer import analyze
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QEventLoop, QTimer


def test_shell_navigation_and_storage(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(THEME)
    window = MainWindow(tmp_path)
    assert window.pages.count() == 11
    for index in range(11):
        window.navigation.setCurrentRow(index)
        assert window.pages.currentIndex() == index
    assert (tmp_path / "history.sqlite3").exists()
    assert window.dataset is None
    window.canvas.add_node("Dataset")
    window.canvas.add_node("Train model")
    window.canvas.graph.edges = [("1", "2")]
    window.canvas.redraw_edges()
    assert window.canvas.graph.order() == ["1", "2"]
    window.show()
    app.processEvents()
    window.grab().save(str(tmp_path / "shell.png"))
    window.close()


def test_project_roundtrip_and_background_worker(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.csv"
    source.write_text("name,value,target\n A,1,0\nb,2,1\n")
    project = tmp_path / "project.json"
    window = MainWindow(tmp_path / "store")
    window.source = str(source)
    window.import_options = {"delimiter": ",", "encoding": "utf-8-sig", "header": True, "decimal": "."}
    window.dataset = DatasetVersions(pd.read_csv(source))
    window.dataset.apply(Transformation("Trim whitespace", ["name"]))
    window.issues = analyze(window.dataset.current)
    window.refresh_data()
    window.target.setCurrentText("target")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(project), ""))
    window.save_project()
    assert project.exists()
    window.dataset = None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(project), ""))
    window.open_project()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if not window.busy else None)
    timer.start(10)
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    timer.stop()
    assert not window.busy
    assert window.dataset.current.name.iloc[0] == "A"
    assert window.target.currentText() == "target"
    assert window.dataset.index == 1
    assert source.read_text().splitlines()[1].startswith(" A")
    window.close()


def test_csv_import_has_no_options_popup_by_default(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog
    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'simple.csv'
    source.write_text('x;group\n1;a\n2;b\n',encoding='utf-8')
    window = MainWindow(tmp_path / 'store')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(source),''))
    monkeypatch.setattr(QDialog,'exec',lambda *a: (_ for _ in ()).throw(AssertionError('Default import must not open options')))
    window.import_csv()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda:loop.quit() if not window.busy else None)
    timer.start(10)
    QTimer.singleShot(10000,loop.quit)
    loop.exec()
    timer.stop()
    assert window.dataset.current.shape == (2,2)
    assert window.target.currentText() == ''
    window.close()
