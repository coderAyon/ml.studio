"""Desktop shell. Expensive operations run on Qt worker threads."""
import json
import logging
import hashlib
import copy
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, QSettings, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog, QLabel, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QStackedWidget, QTableView, QTextBrowser, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import __version__
from .analyzer import analyze, profile
from .data import DatasetVersions, Transformation, load_csv
from .report import export_report
from .storage import Store, user_directory
from .training import ALGORITHMS, REGRESSION, TrainingResult, train, compare, cluster, predict
from .workflow import Graph


class FrameModel(QAbstractTableModel):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def rowCount(self, parent=QModelIndex()):
        return len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and index.isValid():
            value = self.frame.iat[index.row(), index.column()]
            return "—" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.frame.columns[section]) if orientation == Qt.Orientation.Horizontal else str(section + 1)


class WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception as exc:
            logging.exception("Background operation failed")
            self.signals.failed.emit(str(exc))


class NodeItem(QGraphicsRectItem):
    def __init__(self, key, kind, canvas, position):
        super().__init__(0, 0, 190, 74)
        self.key, self.canvas = key, canvas
        self.setBrush(QColor("#ffffff"))
        self.setPen(QPen(QColor("#90a7cc"), 2))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*position)
        from PySide6.QtWidgets import QGraphicsTextItem
        text = QGraphicsTextItem(kind, self)
        text.setPos(12, 18)
        text.setDefaultTextColor(QColor("#223354"))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.canvas.redraw_edges()
        return super().itemChange(change, value)


class WorkflowCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#f0f4fa"))
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.graph = Graph()
        self.items_by_id = {}
        self.lines = []

    def add_node(self, kind, key=None, position=None):
        key = key or str(max([int(k) for k in self.graph.nodes] + [0])+1)
        position = position or (40 + len(self.graph.nodes)*40, 45 + len(self.graph.nodes)*95)
        self.graph.nodes[key] = {"type": kind, "position": position}
        item = NodeItem(key, kind, self, position)
        self.items_by_id[key] = item
        self.scene().addItem(item)

    def redraw_edges(self):
        for line in self.lines:
            self.scene().removeItem(line)
        self.lines = []
        for a, b in self.graph.edges:
            start, end = self.items_by_id[a].sceneBoundingRect(), self.items_by_id[b].sceneBoundingRect()
            line = self.scene().addLine(start.center().x(), start.bottom(), end.center().x(), end.top(), QPen(QColor("#527ce0"), 2))
            line.setZValue(-1)
            self.lines.append(line)

    def connect_selected(self):
        selected = sorted([i for i in self.scene().selectedItems() if isinstance(i, NodeItem)], key=lambda i: i.key)
        if len(selected) != 2:
            raise ValueError("Ctrl-select two nodes. The earlier-created node is the source.")
        edge = (selected[0].key, selected[1].key)
        candidate = Graph(dict(self.graph.nodes), self.graph.edges + [edge])
        # Validate ports locally; incomplete graphs may still be under construction.
        from .workflow import PORTS
        if PORTS[self.graph.nodes[edge[0]]["type"]][1] != PORTS[self.graph.nodes[edge[1]]["type"]][0]:
            raise ValueError("These ports are incompatible.")
        if any(b == edge[1] for a, b in self.graph.edges):
            raise ValueError("This node already has an input connection.")
        self.graph.edges.append(edge)
        self.redraw_edges()

    def snapshot(self):
        for key, item in self.items_by_id.items():
            self.graph.nodes[key]["position"] = [item.pos().x(), item.pos().y()]
        return self.graph

    def restore(self, graph):
        self.scene().clear()
        self.items_by_id, self.lines = {}, []
        self.graph = Graph()
        for key, node in graph.nodes.items():
            self.add_node(node["type"], key, node.get("position", [40, 40]))
            self.graph.nodes[key].update(copy.deepcopy(node))
        self.graph.edges = graph.edges
        self.redraw_edges()


class MainWindow(QMainWindow):
    sections = ["Home", "Workflow Studio", "Dataset Explorer", "Smart Analyzer", "Visual Analytics", "Model Lab", "Deep Learning Lab", "Experiment History", "Models and Predictions", "Reports", "Settings and Help"]

    def __init__(self, directory=None):
        super().__init__()
        self.directory = Path(directory) if directory else user_directory()
        self.store = Store(self.directory)
        self.settings = QSettings("MLStudio", "MLStudio")
        self.pool = QThreadPool(self)
        self.dataset = None
        self.source = None
        self.issues = []
        self.result = None
        self.preparation = None
        self.busy = False
        self.setWindowTitle(f"ML Studio {__version__} · Offline machine learning")
        self.resize(1360, 860)
        shell = QWidget()
        layout = QHBoxLayout(shell)
        self.navigation = QListWidget()
        self.navigation.addItems(self.sections)
        self.navigation.setFixedWidth(220)
        layout.addWidget(self.navigation)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(shell)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.build_pages()
        self.navigation.setCurrentRow(0)
        toolbar = self.addToolBar("Project")
        toolbar.setMovable(False)
        for title, callback, shortcut in [("Import CSV", self.import_csv, "Ctrl+I"), ("Open project", self.open_project, "Ctrl+O"), ("Save project", self.save_project, "Ctrl+S"), ("Export cleaned CSV", self.export_csv, "Ctrl+E")]:
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked=False, fn=callback: self.guard(fn))
            toolbar.addAction(action)
        self.statusBar().showMessage("Ready · All processing stays on this computer")
        self.refresh_history()

    def guard(self, callback):
        try:
            if self.busy:
                raise ValueError("An operation is running. Wait for it to finish.")
            callback()
        except Exception as exc:
            logging.exception("Operation failed")
            self.notify_error(str(exc))

    def notify_error(self, message):
        handler = getattr(self, "error_handler", None)
        if handler:
            handler(message)
        else:
            QMessageBox.warning(self, "ML Studio", message)

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.guard(callback))
        layout.addWidget(button)
        return button

    def page(self, title, subtitle):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        heading = QLabel(title)
        heading.setObjectName("heading")
        layout.addWidget(heading)
        label = QLabel(subtitle)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.pages.addWidget(widget)
        return layout

    def build_pages(self):
        home = self.page("Your local ML workspace", "Explore datasets, review quality findings, and train models with no cloud account.")
        self.button("Import a CSV dataset", self.import_csv, home)
        self.button("Open a saved project", self.open_project, home)
        self.home_info = QTextBrowser()
        home.addWidget(self.home_info)

        flow = self.page("Workflow Studio", "Add nodes, move them on the canvas, then Ctrl-select two and connect. Model settings come from Model Lab. Available nodes are listed below.")
        bar = QHBoxLayout()
        self.node_type = QComboBox()
        self.node_type.addItems(["Dataset", "Remove duplicates", "Train model"])
        bar.addWidget(self.node_type)
        self.button("Add node", lambda: self.canvas.add_node(self.node_type.currentText()), bar)
        self.button("Connect selected", lambda: self.canvas.connect_selected(), bar)
        self.button("Validate", self.validate_graph, bar)
        self.button("Run", self.run_graph, bar)
        self.button("+", lambda: self.canvas.scale(1.2, 1.2), bar)
        self.button("−", lambda: self.canvas.scale(1/1.2, 1/1.2), bar)
        flow.addLayout(bar)
        self.canvas = WorkflowCanvas()
        flow.addWidget(self.canvas)

        explorer = self.page("Dataset Explorer", "Virtual table shows the active dataset version. Original input files are preserved.")
        self.overview = QLabel("Import a CSV to begin.")
        explorer.addWidget(self.overview)
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        explorer.addWidget(self.table)
        self.schema = QTextBrowser()
        self.schema.setMaximumHeight(180)
        explorer.addWidget(self.schema)

        analyzer = self.page("Smart Dataset Analyzer", "Deterministic findings with evidence. Potential issues require domain review.")
        self.button("Analyze dataset", self.analyze_current, analyzer)
        self.issue_list = QListWidget()
        self.issue_list.currentRowChanged.connect(self.show_issue)
        analyzer.addWidget(self.issue_list)
        self.issue_detail = QTextBrowser()
        analyzer.addWidget(self.issue_detail)
        cleaning = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItems(["Fill missing", "Remove duplicates", "Trim whitespace", "Lowercase labels", "Convert to numeric", "Drop columns"])
        self.clean_column = QComboBox()
        cleaning.addWidget(self.operation)
        cleaning.addWidget(self.clean_column)
        self.button("Preview…", self.clean_preview, cleaning)
        self.button("Undo", lambda: self.change_version(False), cleaning)
        self.button("Redo", lambda: self.change_version(True), cleaning)
        analyzer.addLayout(cleaning)

        charts = self.page("Visual Analytics", "Charts use up to the first 10,000 rows, explicitly labeled. Select numeric columns for numeric charts.")
        options = QHBoxLayout()
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Histogram", "Box plot", "Scatter", "Correlation", "Missing values", "Category frequencies"])
        self.chart_x, self.chart_y = QComboBox(), QComboBox()
        for widget in [self.chart_type, self.chart_x, self.chart_y]:
            options.addWidget(widget)
        self.button("Draw chart", self.draw_chart, options)
        charts.addLayout(options)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.plot = FigureCanvasQTAgg(self.figure)
        charts.addWidget(self.plot)

        lab = self.page("Model Lab", "Preprocessing is fitted on training rows only. Classification uses a stratified 75/25 holdout, seed 42. Models use library defaults.")
        form = QFormLayout()
        self.target, self.algorithm = QComboBox(), QComboBox()
        self.algorithm.addItems(ALGORITHMS)
        form.addRow("Target", self.target)
        form.addRow("Algorithm", self.algorithm)
        lab.addLayout(form)
        self.button("Train and evaluate", self.train_model, lab)
        self.button("Compare all algorithms for selected task", self.compare_models, lab)
        clustering = QHBoxLayout()
        self.clustering_algorithm = QComboBox()
        self.clustering_algorithm.addItems(["K-Means", "DBSCAN", "Hierarchical", "PCA"])
        clustering.addWidget(self.clustering_algorithm)
        self.button("Run exploratory clustering / PCA", self.cluster_data, clustering)
        lab.addLayout(clustering)
        self.metrics = QTextBrowser()
        lab.addWidget(self.metrics)
        deep = self.page("Deep Learning Lab", "Unavailable in this development release. PyTorch training and model export are not yet implemented.")
        deep.addStretch()
        history = self.page("Experiment History", "Actual locally stored experiments; timestamps are stored in UTC.")
        self.history = QTableView()
        history.addWidget(self.history)
        models = self.page("Models and Predictions", "The most recently trained pipeline can predict compatible CSV data during this session. Persistent model loading is not available yet.")
        self.button("Predict CSV with current model", self.predict_csv, models)
        self.model_info = QTextBrowser()
        models.addWidget(self.model_info)
        reports = self.page("Reports", "Export actual findings, transformations and current evaluation results to a self-contained HTML report.")
        self.button("Export HTML report", self.report, reports)
        self.button("Export transformation log", self.export_log, reports)
        reports.addStretch()
        settings = self.page("Settings and Help", f"ML Studio {__version__}")
        info = QTextBrowser()
        info.setPlainText(f"Privacy: All data loading, analysis, training and exports run locally. No telemetry, cloud authentication, runtime downloads or external APIs.\n\nLocal metadata: {self.directory}\n\nKeyboard: Ctrl+I import; Ctrl+O open; Ctrl+S save project; Ctrl+E export CSV.\n\nCSV import supports delimiter, encoding, header and decimal overrides. Use analyzer previews before applying changes. Cleaning is exploratory: imputation before evaluation can introduce leakage. Training independently fits its own preprocessing on training rows.\n\nProject files reference the original CSV and store cleaning operations and workflows. Keep the CSV available. Supported project schema: 1.\n\nSee README.md and DEVELOPMENT_STATUS.md for current limitations and manual test instructions.")
        settings.addWidget(info)

    def background(self, function, complete, message):
        self.busy = True
        self.statusBar().showMessage(message)
        worker = Worker(function)
        self.active_worker = worker

        def done(value):
            self.busy = False
            self.statusBar().showMessage("Ready · Operation completed")
            self.guard(lambda: complete(value))

        def failed(message):
            self.busy = False
            self.statusBar().showMessage("Operation failed")
            self.notify_error(message)

        worker.signals.done.connect(done)
        worker.signals.failed.connect(failed)
        self.pool.start(worker)

    def require_data(self):
        if self.dataset is None:
            raise ValueError("Import a dataset first.")
        return self.dataset.current

    def import_csv(self, advanced=False):
        path, _ = QFileDialog.getOpenFileName(self, "Import dataset", "", "CSV and text (*.csv *.tsv *.txt);;All files (*)")
        if not path:
            return
        options = {"delimiter": None, "encoding": "utf-8-sig", "header": True, "decimal": "."}
        if advanced:
            dialog = QDialog(self)
            dialog.setWindowTitle("CSV import options")
            form = QFormLayout(dialog)
            delimiter, encoding, header, decimal = QComboBox(), QComboBox(), QComboBox(), QComboBox()
            delimiter.addItems(["Auto", ",", ";", "Tab", "|"])
            encoding.addItems(["utf-8-sig", "cp1252", "latin-1", "utf-16"])
            header.addItems(["First row contains names", "No header"])
            decimal.addItems([".", ","])
            for name, widget in [("Delimiter", delimiter), ("Encoding", encoding), ("Header", header), ("Decimal", decimal)]:
                form.addRow(name, widget)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            options = {"delimiter": {"Auto": None, "Tab": "\t"}.get(delimiter.currentText(), delimiter.currentText()), "encoding": encoding.currentText(), "header": header.currentIndex() == 0, "decimal": decimal.currentText()}
        def loaded(value):
            frame, issues = value
            self.source, self.import_options = path, options
            self.active_project_path = None
            self.preparation = None
            self.dataset, self.issues, self.result = DatasetVersions(frame), issues, None
            self.refresh_data()
            self.navigation.setCurrentRow(2)

        def read():
            frame = load_csv(path, **options)
            return frame, analyze(frame)
        self.background(read, loaded, "Loading and analyzing CSV…")

    def refresh_data(self):
        frame = self.require_data()
        self.table.setModel(FrameModel(frame))
        p = profile(frame)
        self.overview.setText(f"{Path(self.source).name}  ·  {p['rows']:,} rows × {p['columns']} columns  ·  {p['bytes']/1024**2:.2f} MB  ·  Version {self.dataset.index}")
        self.schema.setPlainText(json.dumps(p["fields"], indent=2))
        for combo in [self.target, self.clean_column, self.chart_x, self.chart_y]:
            old = combo.currentText()
            combo.clear()
            if combo is self.target:
                combo.addItem("")
            combo.addItems(frame.columns)
            if old in frame:
                combo.setCurrentText(old)
        self.issue_list.clear()
        self.issue_list.addItems([f"{i.severity} · {i.title} · {', '.join(i.columns)} ({i.count})" for i in self.issues])
        self.issue_detail.clear()
        if self.result is None:
            self.metrics.clear()
            self.model_info.setPlainText("No current trained model. Train in Model Lab.")

    def analyze_current(self):
        frame = self.require_data()
        self.background(lambda: analyze(frame), self.analysis_done, "Analyzing dataset…")

    def analysis_done(self, issues):
        self.issues = issues
        self.refresh_data()

    def show_issue(self, row):
        if 0 <= row < len(self.issues):
            self.issue_detail.setPlainText(json.dumps(asdict(self.issues[row]), indent=2, default=str))

    def clean_preview(self):
        frame = self.require_data()
        operation = Transformation(self.operation.currentText(), [self.clean_column.currentText()])
        proposed = operation.preview(frame)
        dialog = QDialog(self)
        dialog.setWindowTitle("Review transformation · Original remains preserved")
        dialog.resize(900, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"{operation.operation}: {len(frame)} → {len(proposed)} rows. Preview: first 100 proposed rows.\nImputation here is exploratory; use original data for unbiased model evaluation."))
        comparison = QSplitter()
        for title, data in [("Before", frame), ("After", proposed)]:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.addWidget(QLabel(title))
            table = QTableView()
            table.setModel(FrameModel(data.head(100)))
            panel_layout.addWidget(table)
            comparison.addWidget(panel)
        layout.addWidget(comparison)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.dataset.apply(operation)
            self.result = None
            self.analyze_current()

    def change_version(self, redo):
        self.require_data()
        self.dataset.redo() if redo else self.dataset.undo()
        self.result = None
        self.analyze_current()

    def export_csv(self):
        frame = self.require_data()
        path, _ = QFileDialog.getSaveFileName(self, "Save cleaned copy", "cleaned.csv", "CSV (*.csv)")
        if path:
            self.protect_output(path)
            frame.to_csv(path, index=False)

    def draw_chart(self):
        frame = self.require_data().head(10000)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        kind, x, y = self.chart_type.currentText(), self.chart_x.currentText(), self.chart_y.currentText()
        if kind == "Histogram":
            ax.hist(pd.to_numeric(frame[x], errors="raise").dropna(), bins=30, color="#527ce0")
            ax.set_xlabel(x)
        elif kind == "Box plot":
            ax.boxplot(pd.to_numeric(frame[x], errors="raise").dropna())
            ax.set_ylabel(x)
        elif kind == "Scatter":
            ax.scatter(pd.to_numeric(frame[x], errors="raise"), pd.to_numeric(frame[y], errors="raise"), alpha=.5)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif kind == "Correlation":
            corr = frame.select_dtypes(include="number").corr()
            if corr.empty:
                raise ValueError("No numeric columns for correlation.")
            image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(corr)), corr.columns, rotation=60)
            ax.set_yticks(range(len(corr)), corr.columns)
            self.figure.colorbar(image, ax=ax)
        elif kind == "Missing values":
            frame.isna().sum().plot.bar(ax=ax, color="#d7a143")
        else:
            frame[x].value_counts().head(20).plot.bar(ax=ax, color="#527ce0")
        ax.set_title(f"{kind} · first {len(frame):,} rows")
        self.plot.draw()

    def train_model(self):
        return self.train_with_split(.25)

    def train_with_split(self, test_size=.25,parameters=None,tune=False):
        frame = self.require_data()
        from .training_prep import assert_evaluation_ready
        assert_evaluation_ready(self.dataset)
        target, algorithm = self.target.currentText(), self.algorithm.currentText()
        self.result = None
        self.metrics.clear()
        self.model_info.clear()
        preparation=copy.deepcopy(self.preparation)
        self.background(lambda: train(frame, target, algorithm,test_size=test_size,preparation=preparation,parameters=parameters,tune=tune), self.training_done, "Training and testing model locally…")

    def training_done(self, result):
        self.result = result
        payload = {"metrics": result.metrics, "metadata": result.metadata}
        self.store.add(result.metadata["algorithm"], payload)
        text = json.dumps(payload, indent=2)
        self.metrics.setPlainText(text)
        self.model_info.setPlainText(text)
        self.refresh_history()

    def compare_models(self):
        frame = self.require_data()
        from .training_prep import assert_evaluation_ready
        assert_evaluation_ready(self.dataset)
        regression = self.algorithm.currentText() in REGRESSION
        algorithms = [a for a in ALGORITHMS if (a in REGRESSION) == regression]
        target = self.target.currentText()
        self.result = None
        self.model_info.clear()
        def complete(results):
            for result in results:
                self.store.add(result.metadata["algorithm"], {"metrics": result.metrics, "metadata": result.metadata})
            self.metrics.setPlainText(json.dumps([{ "algorithm": r.metadata["algorithm"], "metrics": r.metrics, "training_seconds": r.metadata["seconds"]} for r in results], indent=2) + "\n\nIdentical holdout splits. This is exploratory comparison; use independent final data after selection. No winner is automatically selected.")
            self.refresh_history()
        preparation=copy.deepcopy(self.preparation)
        self.background(lambda: compare(frame, target, algorithms,preparation=preparation), complete, "Comparing models on identical splits…")

    def cluster_data(self):
        frame = self.require_data()
        algorithm = self.clustering_algorithm.currentText()
        def complete(result):
            self.store.add(algorithm, result)
            self.metrics.setPlainText(json.dumps(result, indent=2))
            self.refresh_history()
        self.background(lambda: cluster(frame, algorithm), complete, "Running exploratory clustering…")

    def refresh_history(self):
        rows = self.store.history()
        self.history.setModel(FrameModel(pd.DataFrame(rows, columns=["Created (UTC)", "Algorithm", "Results"])))
        self.home_info.setPlainText(f"ML Studio {__version__}\n\nStored experiments: {len(rows)}\nStorage: {self.directory}\n\nImport a CSV to begin. No example data is loaded automatically.\n\nCurrent release: dataset analysis, safe cleaning previews, charts, holdout model evaluation and a basic workflow canvas. See Development Status for remaining features.")

    def validate_graph(self):
        self.canvas.graph.order()
        self.statusBar().showMessage("Workflow is valid")

    def run_graph(self):
        frame = self.require_data()
        graph = copy.deepcopy(self.canvas.snapshot())
        target, algorithm = self.target.currentText(), self.algorithm.currentText()
        from .training_prep import assert_evaluation_ready
        assert_evaluation_ready(self.dataset)
        self.result = None
        self.metrics.clear()
        self.model_info.clear()
        def complete(results):
            for value in results.values():
                if isinstance(value, TrainingResult):
                    self.training_done(value)
                elif isinstance(value,list):
                    for result in value:
                        if isinstance(result,TrainingResult):self.store.add(result.metadata['algorithm'],{'metrics':result.metrics,'metadata':result.metadata})
                    self.refresh_history()
            self.statusBar().showMessage(f"Workflow completed: {len(results)} nodes")
        preparation=copy.deepcopy(self.preparation)
        self.background(lambda: graph.execute(frame, target, algorithm,preparation=preparation), complete, "Executing workflow…")

    def save_project(self, path=None):
        self.require_data()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "project.mlstudio.json", "ML Studio (*.json)")
        if path:
            if Path(path).resolve() in {Path(self.source).resolve(), self.store.path.resolve()}:
                raise ValueError("Choose a new project path to preserve the source CSV.")
            graph = self.canvas.snapshot()
            payload = {"version": 1, "source": str(Path(self.source).resolve()), "fingerprint": hashlib.sha256(Path(self.source).read_bytes()).hexdigest() if Path(self.source).is_file() else None, "import": self.import_options, "operations": self.dataset.log[:self.dataset.index], "workflow": {"nodes": graph.nodes, "edges": graph.edges}, "target": self.target.currentText(), "algorithm": self.algorithm.currentText()}
            payload["preparation"] = self.preparation
            from .project_data import pack_frame
            payload['embedded_source']=pack_frame(self.dataset.versions[0])
            from uuid import uuid4
            temporary = Path(path).with_name(Path(path).name+"."+uuid4().hex+".tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self.settings.setValue("lastProject", path)
            self.active_project_path = path

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "ML Studio (*.json)")
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("operations"), list):
            raise ValueError("Unsupported or corrupted project schema.")
        graph = Graph(**payload["workflow"])
        if graph.nodes:
            graph.order(allow_disconnected=True)
        def read():
            embedded=payload.get('embedded_source')
            if embedded:
                from .project_data import unpack_frame
                frame=unpack_frame(embedded)
            else:
                if payload.get("fingerprint") != hashlib.sha256(Path(payload["source"]).read_bytes()).hexdigest():
                    raise ValueError("The source CSV has changed since this project was saved. Import the changed file as a new dataset.")
                frame = load_csv(payload["source"], **payload["import"])
            versions = DatasetVersions(frame)
            for operation in payload["operations"]:
                versions.apply(Transformation(**operation))
            from .training_prep import validate_preparation
            preparation=validate_preparation(payload.get("preparation"),versions.current.columns,payload.get("target"))
            return versions, analyze(versions.current), preparation
        def loaded(value):
            self.dataset, self.issues, self.preparation = value
            self.active_project_path = path
            self.source, self.import_options = payload["source"], payload["import"]
            self.result = None
            self.canvas.restore(graph)
            self.refresh_data()
            self.target.setCurrentText(payload.get("target", ""))
            self.algorithm.setCurrentText(payload.get("algorithm", ""))
        self.background(read, loaded, "Reopening project…")

    def report(self):
        frame = self.require_data()
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "report.html", "HTML (*.html)")
        if path:
            self.protect_output(path)
            export_report(path, frame, self.issues, self.dataset.log[:self.dataset.index], {"metrics": self.result.metrics, "metadata": self.result.metadata} if self.result else {})

    def export_log(self):
        self.require_data()
        path, _ = QFileDialog.getSaveFileName(self, "Export transformation log", "transformations.json", "JSON (*.json)")
        if path:
            self.protect_output(path)
            Path(path).write_text(json.dumps(self.dataset.log[:self.dataset.index], indent=2), encoding="utf-8")

    def protect_output(self, path):
        protected=[self.source, getattr(self,"active_project_path",None), self.store.path]
        if Path(path).resolve() in [Path(p).resolve() for p in protected if p]:
            raise ValueError("Choose a separate output file to preserve your dataset, saved project and history.")

    def save_model(self):
        from .model_io import save_model
        if self.result is None:raise ValueError('Train or load a model first.')
        path,_=QFileDialog.getSaveFileName(self,'Save trained model','trained-model.mlmodel','ML Studio model (*.mlmodel)')
        if not path:return
        self.protect_output(path)
        save_model(path,self.result)
        self.statusBar().showMessage('Trained pipeline saved. It can predict after reopening.')

    def load_model(self):
        from .model_io import load_model
        path,_=QFileDialog.getOpenFileName(self,'Open trained model','','ML Studio model (*.mlmodel)')
        if not path:return
        def complete(result):
            self.result=result
            self.model_info.setPlainText(json.dumps(result.metadata,indent=2))
            self.statusBar().showMessage('Trained model loaded. Ready to predict a compatible CSV.')
        self.background(lambda:load_model(path),complete,'Loading saved prediction pipeline…')

    def predict_csv(self,options=None):
        if self.result is None:
            raise ValueError("Train a model first.")
        path, _ = QFileDialog.getOpenFileName(self, "Prediction input", "", "CSV (*.csv)")
        if not path:
            return
        data = load_csv(path,**(options or getattr(self,'import_options',{}) or {}))
        expected = self.result.metadata["features"]
        if set(data.columns) != set(expected):
            raise ValueError(f"Input columns must match exactly. Expected: {expected}")
        predictions = predict(self.result, data)
        output, _ = QFileDialog.getSaveFileName(self, "Save predictions", "predictions.csv", "CSV (*.csv)")
        if output:
            self.protect_output(output)
            if Path(output).resolve() in [Path(p).resolve() for p in [path,self.source] if p]:
                raise ValueError("Choose a new output path to preserve input data.")
            name = "prediction"
            while name in data:
                name = "_"+name
            data.assign(**{name:predictions}).to_csv(output, index=False)

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "Operation running", "Wait for the current operation to finish before closing.")
            event.ignore()
        else:
            self.settings.setValue("geometry", self.saveGeometry())
            event.accept()


THEME = """
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #24324b; }
QMainWindow, QStackedWidget { background: #f7f9fc; }
QLabel#heading { font-size: 27px; font-weight: 650; margin-bottom: 8px; }
QListWidget, QTextBrowser, QTableView, QComboBox { background: white; border: 1px solid #dde4ef; border-radius: 6px; padding: 6px; }
QListWidget::item { padding: 11px 8px; }
QListWidget::item:selected { background: #e4edff; color: #244d9c; border-radius: 4px; }
QPushButton { background: #365fc2; color: white; border: 0; padding: 9px 14px; border-radius: 5px; }
QPushButton:hover { background: #244aab; }
QPushButton:disabled { background: #c5cddd; }
QToolBar { background: white; border-bottom: 1px solid #dde4ef; spacing: 12px; padding: 8px; }
QHeaderView::section { background: #edf2fa; padding: 8px; border: 0; }
"""


def main():
    directory = user_directory()
    logging.basicConfig(filename=directory / "mlstudio.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .desktop import ModernWindow
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(THEME)
    if '--flow-smoke' in sys.argv:
        from .flow_smoke import run
        index=sys.argv.index('--flow-smoke')
        try:run(sys.argv[index+1])
        except Exception:
            import traceback
            output=Path(sys.argv[index+1]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');sys.exit(1)
        return
    if '--student-upgrade-smoke' in sys.argv:
        from .student_upgrade_smoke import run
        index=sys.argv.index('--student-upgrade-smoke')
        try:run(sys.argv[index+1])
        except Exception:
            import traceback
            output=Path(sys.argv[index+1]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');sys.exit(1)
        return
    if '--audit-fix-smoke' in sys.argv:
        from .audit_fix_smoke import run
        index=sys.argv.index('--audit-fix-smoke')
        try:run(sys.argv[index+1])
        except Exception:
            import traceback
            output=Path(sys.argv[index+1]);output.mkdir(parents=True,exist_ok=True)
            (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            sys.exit(1)
        return
    if "--smoke-test" in sys.argv:
        window = MainWindow(directory)
        from .smoke import run
        index = sys.argv.index("--smoke-test")
        if len(sys.argv) <= index + 1:
            raise ValueError("Usage: MLStudio --smoke-test OUTPUT_DIRECTORY")
        run(window, app, sys.argv[index + 1])
        return
    if "--research-smoke" in sys.argv:
        from .research_smoke import run
        index = sys.argv.index("--research-smoke")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        run(window, app, sys.argv[index + 1])
        return
    if "--pastel-smoke" in sys.argv:
        from .pastel_smoke import run
        index = sys.argv.index("--pastel-smoke")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        run(window, app, sys.argv[index + 1])
        return
    if "--beginner-smoke" in sys.argv:
        from .beginner_smoke import run
        index = sys.argv.index("--beginner-smoke")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        try:
            run(window, app, sys.argv[index + 1])
        except Exception:
            import traceback
            output = Path(sys.argv[index + 1])
            output.mkdir(parents=True,exist_ok=True)
            (output / "failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
            raise
        return
    if "--contest-audit" in sys.argv:
        from .contest_audit import run
        index = sys.argv.index("--contest-audit")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        run(window, app, sys.argv[index + 1])
        return
    if "--appearance-smoke" in sys.argv:
        from .appearance_smoke import run
        index = sys.argv.index("--appearance-smoke")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        run(window, app, sys.argv[index + 1])
        return
    if "--ui-smoke" in sys.argv:
        from .ui_smoke import run
        index = sys.argv.index("--ui-smoke")
        window = ModernWindow(Path(sys.argv[index + 1]) / "isolated-store")
        run(window, app, sys.argv[index + 1])
        return
    window = ModernWindow(directory)
    geometry = window.controller.settings.value("modernGeometry")
    if geometry:
        window.restoreGeometry(geometry)
    window.show()
    sys.exit(app.exec())
