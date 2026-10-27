"""Offline embedded web presentation over the existing desktop application engine."""
import base64
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QApplication
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestInterceptor, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView

from . import __version__
from .analyzer import profile
from .app import MainWindow, THEME
from .data import Transformation
from .workflow import Graph, PORTS
from .guidance import visual_summary, preprocessing_plan, explain_comparison
from .training import REGRESSION
from .research_controller import ResearchCoordinator

ASSETS = Path(__file__).parent / "web"


def allowed_url(url: QUrl) -> bool:
    """Only packaged assets and inline resources may be loaded by the renderer."""
    if url.scheme() in {"qrc", "data", "blob", "about"}:
        return True
    if url.scheme() == "file":
        try:
            return Path(url.toLocalFile()).resolve().is_relative_to(ASSETS.resolve())
        except (ValueError, OSError):
            return False
    return False


class OfflineRequests(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        if not allowed_url(info.requestUrl()):
            info.block(True)


class OfflinePage(QWebEnginePage):
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        return allowed_url(url)

    def javaScriptConsoleMessage(self, level, message, line, source):
        logging.info("Web UI %s:%s %s", source, line, message)


class StudioBridge(QObject):
    """Explicit command allowlist; the page cannot run Python or read arbitrary files."""
    stateChanged = Signal(str)
    notice = Signal(str)

    def __init__(self, controller: MainWindow, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.chart = ""
        self._chart_key = None
        self.preview = None
        self.table_options = {}
        self.clustering_result = None
        self.clustering_key = None
        self._cache_key = None
        self._cache = None
        self.last_status = "Your workspace is ready."
        self.events = []
        self.comparison_start = None
        self._guidance_key = None
        self._guidance = None
        self.visual_choices = {}
        self.lab_result = self.lab_figure = self.lab_key = None
        self.row_detail = None
        self.visual_review = None
        self.visual_batch = []
        self.visual_results = []
        self.visual_figures = {}
        self.visual_request_key = None
        self.preprocess_key = None
        self.preprocess_detail = self.preprocess_preview = None
        self.training_preview = None
        self.dataset_visual_result = self.dataset_visual_figure = self.dataset_visual_key = None
        controller.error_handler = lambda message:self.announce(message,"error")
        controller.statusBar().messageChanged.connect(self.on_status)
        self.research=ResearchCoordinator(self)
        from .prediction_controller import PredictionWorkspace
        self.predictions=PredictionWorkspace(self)
        from .flow_controller import FlowCoordinator
        self.flow=FlowCoordinator(self)

    def on_status(self, message):
        self.last_status = message
        self.events.append(message)
        self.events = self.events[-15:]
        QTimer.singleShot(0, self.publish)

    def dataset_state(self):
        controller = self.controller
        if controller.dataset is None:
            return None
        key = (id(controller.dataset), controller.dataset.index)
        if key != self._cache_key:
            self.row_detail = None
            self.comparison_start = None
            frame = controller.dataset.current
            summary = profile(frame)
            numeric = frame.select_dtypes(include="number")
            distribution = None
            populated = [c for c in numeric if numeric[c].notna().any()]
            if populated:
                column = populated[0]
                finite = numeric[column].replace([float("inf"),float("-inf")],float("nan")).dropna()
                counts = pd.cut(finite, bins=14, duplicates="drop").value_counts(sort=False) if not finite.empty else pd.Series(dtype=int)
                distribution = {"column": column, "values": counts.tolist()}
            self._cache = {
                **summary, "name": Path(controller.source).name, "source": str(controller.source),
                "version": controller.dataset.index,
                "preview": json.loads(frame.head(100).to_json(orient="split", date_format="iso")),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicates": int(frame.duplicated().sum()),
                "distribution": distribution,
                "transformations": controller.dataset.log[:controller.dataset.index],
                "can_undo": controller.dataset.index > 0,
                "can_redo": controller.dataset.index < len(controller.dataset.versions) - 1,
                "visuals": visual_summary(frame, **self.visual_choices),
            }
            self._cache_key = key
            if self._chart_key != key:self.chart = ""
        result = dict(self._cache)
        if self.table_options:
            frame = controller.dataset.current
            search = str(self.table_options.get("search", ""))
            if search:
                mask = pd.Series(False, index=frame.index)
                for column in frame:
                    mask |= frame[column].astype("string").str.contains(search, case=False, regex=False, na=False)
                frame = frame.loc[mask]
            column = self.table_options.get("sort")
            if column in frame:
                frame = frame.sort_values(column, ascending=not self.table_options.get("descending", False), kind="stable", key=lambda s: s.astype(str) if s.dtype == "object" else s)
            total = len(frame)
            page = min(max(0, int(self.table_options.get("page", 0))), max(0,(total-1)//100))
            result["preview"] = json.loads(frame.iloc[page*100:(page+1)*100].to_json(orient="split", date_format="iso"))
            result["table_page"], result["filtered_rows"] = page, total
        return result

    def snapshot(self):
        controller = self.controller
        request_key = (id(controller.dataset), controller.dataset.index if controller.dataset else None)
        if self.dataset_visual_key != request_key:
            self.dataset_visual_result = self.dataset_visual_figure = None
            self.dataset_visual_key = request_key
        if self.preprocess_key != request_key:
            self.preprocess_detail = self.preprocess_preview = None
            self.training_preview = None
            self.preprocess_key = request_key
        if self.visual_request_key != request_key:
            self.visual_review = None
            self.visual_batch = []
            self.visual_results = []
            self.visual_figures = {}
            self.visual_request_key = request_key
        if self.lab_key != (id(controller.dataset), controller.dataset.index if controller.dataset else None):
            self.lab_result = self.lab_figure = self.lab_key = None
        if self.clustering_key != (id(controller.dataset),controller.dataset.index if controller.dataset else None):
            self.clustering_result = None
        records = [{"created": t, "algorithm": a, "result": json.loads(r)} for t, a, r in controller.store.history()]
        dataset = self.dataset_state()
        target = controller.target.currentText()
        task = "regression" if controller.algorithm.currentText() in REGRESSION else "classification"
        key = (self._cache_key, target, task)
        if dataset and key != self._guidance_key:
            self._guidance = preprocessing_plan(controller.dataset.current, target, task)
            self._guidance_key = key
        comparison = records[:max(0,len(records)-self.comparison_start)] if self.comparison_start is not None else []
        return {
            "version": __version__, "busy": controller.busy, "status": self.last_status,
            "dataset": dataset, "issues": [asdict(issue) for issue in controller.issues],
            "experiments": records[:100], "experiment_count": len(records),
            "model": {"metrics": controller.result.metrics, "metadata": controller.result.metadata} if controller.result else None,
            "target": controller.target.currentText(), "algorithm": controller.algorithm.currentText(),
            "algorithms": [controller.algorithm.itemText(i) for i in range(controller.algorithm.count())],
            "regression_algorithms":sorted(REGRESSION),
            "parameter_specs":self.parameter_specs(),
            "workflow": {"nodes": controller.canvas.snapshot().nodes, "edges": controller.canvas.graph.edges},
            "storage": str(controller.directory), "chart": self.chart,
            "preview": self.preview, "events": self.events,
            "comparison": comparison,
            "guidance": self._guidance if dataset else None,
            "comparison_explanation": explain_comparison(comparison, task),
            "research": self.research.state(),
            "clustering":self.clustering_result,
            "lab": self.lab_result,
            "row_detail": self.row_detail if dataset else None,
            "visual_requests": {"review":self.visual_review,"results":self.visual_results,"batch":self.visual_batch},
            "preprocessing": {"detail":self.preprocess_detail,"preview":self.preprocess_preview,"training_preview":self.training_preview,
                              "preparation":self.training_preparation()},
            "dataset_visuals": self.dataset_visual_result,
            "predictions":self.predictions.state(),
            "flow":self.flow.state(),
            "preferences": {"theme": controller.settings.value("referenceTheme", "pastel"), "reduce_motion": controller.settings.value("referenceReduceMotion",False,type=bool), "autosave": controller.settings.value("referenceAutosave", False, type=bool), "research_rows":controller.settings.value("researchRowBudget",20000,type=int), "next_storage":controller.settings.value("storageRoot", "")},
        }

    @Slot()
    def publish(self):
        try:
            self.stateChanged.emit(json.dumps(self.snapshot(), default=str, allow_nan=False))
        except Exception as exc:
            logging.exception("Could not render workspace state")
            self.notice.emit(json.dumps({"kind": "error", "message": str(exc)}))

    def announce(self, message, kind="success"):
        self.notice.emit(json.dumps({"kind": kind, "message": message}))

    def training_preparation(self):
        from .training_prep import validate_preparation
        return validate_preparation(getattr(self.controller,'preparation',None))

    def parameter_specs(self):
        from .training import ALGORITHMS
        from .model_settings import parameter_specs
        return {a:parameter_specs(a) for a in ALGORITHMS}

    def prediction_options(self,data):
        options=data.get('import',{})
        if not isinstance(options,dict) or set(options)-{'encoding','decimal','delimiter','header'}:raise ValueError('Invalid prediction import settings.')
        if options.get('encoding','utf-8-sig') not in {'utf-8-sig','cp1252','latin-1','utf-16'} or options.get('decimal','.') not in {'.',','} or options.get('delimiter') not in {None,',',';','\t','|'} or type(options.get('header',True)) is not bool:
            raise ValueError('Choose supported CSV import settings.')
        return options

    def remove_dataset(self):
        controller = self.controller
        if controller.dataset is None:
            return
        answer = QMessageBox.question(
            self.parent(), "Remove dataset?",
            "Remove the imported dataset from this workspace? Unsaved cleaning changes "
            "and current model/research results will be cleared. Your original CSV, "
            "saved projects and experiment history will remain on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        controller.dataset = controller.source = controller.result = None
        controller.active_project_path = None
        controller.preparation = None
        controller.import_options = {}
        controller.issues = []
        controller.table.setModel(None)
        for combo in [controller.target, controller.clean_column, controller.chart_x, controller.chart_y]:
            combo.clear()
        for widget in [controller.schema, controller.issue_list, controller.issue_detail,
                       controller.metrics, controller.model_info]:
            widget.clear()
        controller.overview.setText("No dataset imported. Import a CSV to get started.")
        self.chart, self.preview = "", None
        self._chart_key = self._cache_key = self._cache = None
        self._guidance_key = self._guidance = None
        self.clustering_result = self.clustering_key = self.comparison_start = None
        self.visual_choices, self.table_options = {}, {}
        self.lab_result = self.lab_figure = self.lab_key = None
        old_research = self.research
        self.research = ResearchCoordinator(self)
        old_research.deleteLater()
        self.announce("Dataset removed from workspace. Original CSV and saved files preserved.")

    @Slot(str, str)
    def command(self, action, payload="{}"):
        controller = self.controller
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Invalid command payload.")
            if action == "ready":
                self.publish()
                return
            appearance_only=action=="preferences" and bool(data) and set(data)<={"theme","reduce_motion"}
            if controller.busy and not appearance_only:
                raise ValueError("An operation is running. Wait for it to finish.")
            if action.startswith("research_"):
                self.research.command(action,data)
                self.publish()
                return
            if action.startswith('prediction_'):
                self.predictions.command(action,data);self.publish();return
            if action.startswith('flow_'):
                self.flow.command(action,data);self.publish();return
            if action in {"simple_train","simple_compare"}:
                from .training import ALGORITHMS, compare
                from .training_prep import assert_evaluation_ready,validate_preparation
                import copy
                frame=controller.require_data()
                target=data.get('target');task=data.get('task')
                if target not in frame or task not in {'classification','regression'}:
                    raise ValueError("Choose what you want to predict and whether it is a category or number.")
                assert_evaluation_ready(controller.dataset)
                prep=validate_preparation(controller.preparation,frame.columns,target)
                test_size=data.get('test_size',.25)
                if isinstance(test_size,bool) or not isinstance(test_size,(int,float)) or not .1<=test_size<=.4:
                    raise ValueError("Choose a test share between 10% and 40%.")
                algorithm=data.get('algorithm') or ('Ridge' if task=='regression' else 'Logistic Regression')
                if algorithm not in ALGORITHMS or (algorithm in REGRESSION)!=(task=='regression'):
                    raise ValueError("Choose an algorithm that matches your task.")
                controller.target.setCurrentText(target);controller.algorithm.setCurrentText(algorithm)
                if action=='simple_train':
                    self.comparison_start=None
                    controller.train_with_split(test_size=test_size,parameters=data.get('parameters'),tune=data.get('tune',False))
                else:
                    algorithms=data.get('algorithms') or (['Ridge','Decision Tree Regressor','Random Forest Regressor'] if task=='regression' else ['Logistic Regression','Decision Tree','Random Forest'])
                    if not isinstance(algorithms,list) or not 1<=len(algorithms)<=6 or len(set(algorithms))!=len(algorithms) or any(a not in ALGORITHMS or (a in REGRESSION)!=(task=='regression') for a in algorithms):
                        raise ValueError('Choose 1–6 distinct methods for the same task.')
                    self.comparison_start=len(controller.store.history())
                    controller.result=None
                    def complete(results):
                        for result in results:
                            controller.store.add(result.metadata['algorithm'],{'metrics':result.metrics,'metadata':result.metadata})
                        controller.refresh_history()
                    self.research._background(lambda:compare(frame,target,algorithms,preparation=copy.deepcopy(prep),test_size=test_size),
                                              "Three starting methods compared on the same split. Results are exploratory.",complete)
                self.publish()
                return
            methods = {
                "remove_dataset": self.remove_dataset,
                "import": controller.import_csv, "open": controller.open_project,
                "import_advanced": lambda: controller.import_csv(advanced=True),
                "save": controller.save_project, "export_csv": controller.export_csv,
                "analyze": controller.analyze_current, "train": controller.train_model,
                "compare": controller.compare_models, "predict": controller.predict_csv,
                "report": controller.report, "export_log": controller.export_log,
                "save_model":controller.save_model,"load_model":controller.load_model,
                "predict_configured":lambda:controller.predict_csv(options=self.prediction_options(data)),
                "run_workflow": controller.run_graph,
            }
            if action in methods:
                if action == "compare":
                    self.comparison_start = len(controller.store.history())
                elif action=='run_workflow' and any(n['type']=='Compare models' for n in controller.canvas.graph.nodes.values()):
                    self.comparison_start=len(controller.store.history())
                elif action in {"import", "open", "train", "run_workflow"}:
                    self.comparison_start = None
                if action in {"import", "import_advanced", "open"}:
                    self.preview, self.chart = None, ""
                    self.visual_choices = {}
                    self.table_options = {}
                methods[action]()
            elif action == "configure":
                self.training_preview=None
                self.comparison_start = None
                if data.get("target") in [controller.target.itemText(i) for i in range(controller.target.count())]:
                    controller.target.setCurrentText(data["target"])
                if data.get("algorithm") in [controller.algorithm.itemText(i) for i in range(controller.algorithm.count())]:
                    controller.algorithm.setCurrentText(data["algorithm"])
            elif action == "preprocess_inspect":
                self.training_preview=None
                from .preprocessing import inspect_finding
                frame=controller.require_data();self.snapshot()
                issue=next((i for i in controller.issues if i.id==str(data.get('id'))),None)
                if issue is None:raise ValueError("Select a current finding to inspect.")
                self.preprocess_detail=inspect_finding(frame,asdict(issue))
                if controller.target.currentText() in issue.columns:
                    self.preprocess_detail['caution'] += " This is the selected outcome column. Changing it changes the labels you are trying to predict; review the source and domain meaning first."
                self.preprocess_preview=None
            elif action == "preprocess_preview":
                self.training_preview=None
                from .preprocessing import preview_change
                frame=controller.require_data();self.snapshot()
                self.preprocess_preview=preview_change(frame,data.get('operation'),data.get('column',''),str(data.get('value','')))
            elif action == "preprocess_cancel":
                self.preprocess_preview=None
            elif action == "preprocess_apply":
                frame=controller.require_data()
                key=(id(controller.dataset),controller.dataset.index)
                if self.preprocess_key!=key or not self.preprocess_preview:
                    raise ValueError("Preview the change on the current dataset first.")
                operations=self.preprocess_preview['operations']
                proposed=frame
                for op in operations:
                    proposed=Transformation(**op).preview(proposed)
                    if proposed.empty:raise ValueError("Keep at least one row and one column.")
                for op in operations:controller.dataset.apply(Transformation(**op))
                controller.result=None;self.comparison_start=None;self.preview=None
                self.preprocess_detail=self.preprocess_preview=None
                controller.analyze_current()
                self.announce("Cleaning applied as a new dataset version. Undo is available; your source CSV is preserved.")
            elif action == "training_preview":
                from .flow_engine import FlowEngine,Packet
                from .training_prep import validate_preparation
                frame=controller.require_data();self.snapshot()
                if len(frame)>20000 or len(frame.columns)>300:
                    raise ValueError('Preparation preview supports up to 20,000 rows and 300 columns. Use a smaller extract.')
                plan=validate_preparation(data.get('preparation',{}),frame.columns,controller.target.currentText())
                _,result=FlowEngine().prepared(Packet(frame,plan),controller.target.currentText(),controller.algorithm.currentText(),data)
                self.training_preview=result['comparison']
            elif action == "training_preparation":
                from .training_prep import validate_preparation
                frame=controller.require_data()
                controller.preparation=validate_preparation(data,frame.columns,controller.target.currentText())
                controller.result=None;self.comparison_start=None
                self.training_preview=None
                self.announce("Training preparation saved. It will be fitted only on training rows; your dataset is unchanged.")
            elif action == "visuals":
                frame = controller.require_data()
                self.visual_choices = {k:v for k,v in data.items() if k in {"x","y","category"} and v in frame}
                self._cache_key = None
            elif action == "inspect_row":
                frame = controller.require_data()
                position = data.get("row")
                if not isinstance(position,int) or isinstance(position,bool) or not 1<=position<=len(frame):
                    raise ValueError("Choose an active row number between 1 and "+str(len(frame))+".")
                row = frame.iloc[position-1]
                values = json.loads(frame.iloc[[position-1]].to_json(orient="records",date_format="iso"))[0]
                self.row_detail = {"position":position,"index":str(frame.index[position-1]),
                    "missing":int(row.isna().sum()),"present":int(row.notna().sum()),
                    "values":[{"column":str(col),"value":values[col],"type":str(frame[col].dtype),
                               "missing":bool(pd.isna(row[col])),
                               "infinite":bool(pd.api.types.is_numeric_dtype(frame[col]) and not pd.isna(row[col]) and row[col] in [float('inf'),float('-inf')])} for col in frame]}
            elif action == "table_query":
                self.table_options = {k:v for k,v in data.items() if k in {"search", "sort", "descending", "page"}}
            elif action == "preferences":
                if "theme" in data and data["theme"] not in {"pastel","contrast","dark"}:
                    raise ValueError("Choose Light, Dark or High contrast light.")
                if "reduce_motion" in data and not isinstance(data["reduce_motion"],bool):
                    raise ValueError("Reduce motion must be on or off.")
                if "research_rows" in data:
                    limit = int(data["research_rows"])
                    if not 100<=limit<=20000:
                        raise ValueError("Research row budget must be between 100 and 20,000.")
                    controller.settings.setValue("researchRowBudget", limit)
                if data.get("theme") in {"pastel", "contrast", "dark"}:
                    controller.settings.setValue("referenceTheme", data["theme"])
                    owner=self.parent()
                    if hasattr(owner,"update_native_theme"):owner.update_native_theme(data["theme"])
                    if self.chart:self.render_chart()
                    if self.lab_figure is not None:self.render_lab()
                    for result in self.visual_results:self.render_visual_result(result)
                    if self.dataset_visual_figure is not None:self.render_dataset_visual()
                if "reduce_motion" in data:
                    controller.settings.setValue("referenceReduceMotion",data["reduce_motion"])
                if "autosave" in data:
                    controller.settings.setValue("referenceAutosave", bool(data["autosave"]))
            elif action == "storage_location":
                location = QFileDialog.getExistingDirectory(self.parent(), "Local storage location (takes effect after restart)", str(controller.directory))
                if location:
                    controller.settings.setValue("storageRoot", location)
                    self.announce("New local storage location saved for next launch. Existing files remain in their current folder.")
            elif action == "autosave":
                path = getattr(controller, "active_project_path", None)
                if path and controller.dataset is not None:
                    controller.save_project(path)
                    self.announce("Project autosaved.")
            elif action == "health_export":
                import html
                from PySide6.QtGui import QTextDocument, QPdfWriter
                frame = controller.require_data()
                path, _ = QFileDialog.getSaveFileName(self.parent(), "Export dataset health report", "dataset-health."+data.get("format", "html"), "Reports (*.html *.csv *.pdf)")
                if path:
                    controller.protect_output(path)
                    if Path(path).resolve() == Path(controller.source).resolve():
                        raise ValueError("Choose a new output path to preserve the original dataset.")
                    records = [asdict(i) for i in controller.issues]
                    if path.lower().endswith(".csv"):
                        pd.DataFrame(records).to_csv(path, index=False)
                    else:
                        completeness = 100*(1-frame.isna().sum().sum()/max(1,frame.size))
                        uniqueness = 100*(1-frame.duplicated().sum()/max(1,len(frame)))
                        body = "<h1>Dataset health report</h1><p>"+html.escape(str(controller.source))+"</p><p>Dataset version: "+str(controller.dataset.index)+f"</p><p>Health indicator: {(completeness+uniqueness)/2:.2f}/100. Mean of completeness ({completeness:.2f}%) and exact-row uniqueness ({uniqueness:.2f}%). This is not a measure of predictive accuracy or validity.</p>"+pd.DataFrame(records).to_html(index=False, escape=True)
                        if path.lower().endswith(".pdf"):
                            document = QTextDocument()
                            document.setHtml(body)
                            writer = QPdfWriter(path)
                            document.print_(writer)
                        else:
                            Path(path).write_text("<!doctype html><meta charset='utf-8'>"+body, encoding="utf-8")
                    self.announce("Dataset health report exported.")
            elif action == "bulk_preview":
                frame = controller.require_data()
                operations = data.get("operations", [])
                if not operations or len(operations)>100:
                    raise ValueError("Select between 1 and 100 cleaning steps.")
                proposed = frame
                for operation in operations:
                    proposed = Transformation(**operation).preview(proposed)
                    if proposed.empty:raise ValueError("This cleaning plan would remove every row or column. Keep at least one row and column.")
                self.preview = {"operation": {"operation":"Reviewed cleaning plan", "columns":[]}, "operations":operations, "before_rows":len(frame), "after_rows":len(proposed), "before":json.loads(frame.head(8).to_json(orient="split",date_format="iso")), "after":json.loads(proposed.head(8).to_json(orient="split",date_format="iso"))}
                self.preview["quality"] = {"before_missing":int(frame.isna().sum().sum()), "after_missing":int(proposed.isna().sum().sum()), "before_duplicates":int(frame.duplicated().sum()), "after_duplicates":int(proposed.duplicated().sum())}
            elif action == "task":
                self.comparison_start = None
                controller.algorithm.setCurrentText("Ridge" if data.get("task")=="regression" else "Logistic Regression")
            elif action == "undo":
                controller.change_version(False)
                self.chart = ""
            elif action == "redo":
                controller.change_version(True)
                self.chart = ""
            elif action == "preview":
                frame = controller.require_data()
                transform = Transformation(data["operation"], [data["column"]], str(data.get("value", "")))
                proposed = transform.preview(frame)
                self.preview = {"operation": asdict(transform), "before_rows": len(frame), "after_rows": len(proposed), "before": json.loads(frame.head(8).to_json(orient="split")), "after": json.loads(proposed.head(8).to_json(orient="split"))}
                self.preview["quality"] = {"before_missing":int(frame.isna().sum().sum()), "after_missing":int(proposed.isna().sum().sum()), "before_duplicates":int(frame.duplicated().sum()), "after_duplicates":int(proposed.duplicated().sum())}
            elif action == "cancel_preview":
                self.preview = None
            elif action == "apply_preview":
                if not self.preview:
                    raise ValueError("Preview a transformation before applying it.")
                operations = self.preview.get("operations", [self.preview["operation"]])
                # Validate the whole plan before creating any dataset versions.
                proposed = controller.require_data()
                for operation in operations:
                    proposed = Transformation(**operation).preview(proposed)
                    if proposed.empty:raise ValueError("This change would remove every row or column. Keep at least one row and column.")
                for operation in operations:
                    controller.dataset.apply(Transformation(**operation))
                controller.result = None
                self.preview, self.chart = None, ""
                controller.analyze_current()
                self.announce("New dataset version created. Your original file is preserved.")
            elif action == "dataset_visual_chart":
                from .dataset_visuals import build_chart
                frame=controller.require_data();self.snapshot()
                result,figure=build_chart(frame,data)
                result['dataset']={'name':Path(controller.source).name,'version':controller.dataset.index}
                old_result,old_figure=self.dataset_visual_result,self.dataset_visual_figure
                try:
                    self.dataset_visual_result,self.dataset_visual_figure=result,figure
                    self.render_dataset_visual()
                except Exception:
                    self.dataset_visual_result,self.dataset_visual_figure=old_result,old_figure
                    raise
            elif action in {'dataset_visual_select','dataset_visual_selection_export','dataset_visual_selection_apply'}:
                from .dataset_visuals import select_region
                frame=controller.require_data();self.snapshot();result=self.dataset_visual_result
                if result is None or result['kind']!='scatter':raise ValueError('Create a scatter plot first.')
                if action=='dataset_visual_select':
                    self.dataset_visual_result['selection']=select_region(frame,{**data,'columns':result['config']['columns']})
                else:
                    selection=result.get('selection')
                    if not selection or not selection['count']:raise ValueError('Select some points first.')
                    if action.endswith('export'):
                        path,_=QFileDialog.getSaveFileName(self.parent(),'Save selected rows','selected-rows.csv','CSV (*.csv)')
                        if path:controller.protect_output(path);frame.iloc[selection['positions']].to_csv(path,index=False)
                    else:
                        controller.dataset.apply(Transformation('Select row positions',[],json.dumps(selection['positions'])))
                        controller.result=None;controller.analyze_current()
                        self.announce('Selected records are now the active version. Undo restores the previous dataset.')
            elif action == "dataset_visual_export":
                from .dataset_visuals import detect_outliers
                frame=controller.require_data();self.snapshot()
                result=self.dataset_visual_result
                if result is None:raise ValueError('Create a dataset chart before exporting it.')
                format=data.get('format','png')
                if format not in {'png','svg','csv','outliers','correlation_pairs'}:raise ValueError('Choose a supported chart export.')
                suffix='csv' if format in {'outliers','correlation_pairs'} else format
                path,_=QFileDialog.getSaveFileName(self.parent(),'Export dataset visualization','dataset-visual.'+suffix,suffix.upper()+' files (*.'+suffix+')')
                if path:
                    controller.protect_output(path)
                    if format in {'png','svg'}:
                        self.dataset_visual_figure.savefig(path,format=format,dpi=180,bbox_inches='tight',facecolor=self.dataset_visual_figure.get_facecolor())
                    elif format=='outliers':
                        columns=result['config'].get('columns',[])
                        mask,_=detect_outliers(frame,columns,result['config'].get('outlier_method','iqr'))
                        export=frame.iloc[[i for i,value in enumerate(mask) if value]].copy()
                        import numpy as np
                        name='active_row'
                        while name in export:name='_'+name
                        export.insert(0,name,(np.flatnonzero(mask)+1).tolist())
                        export.to_csv(path,index=False)
                    else:
                        table=result['pairs'] if format=='correlation_pairs' else result['table']
                        if table is None:raise ValueError('Create a correlation chart first.')
                        pd.DataFrame(table['data'],columns=table['columns']).to_csv(path,index=False)
            elif action == "visual_prepare":
                from .visual_requests import parse_request
                controller.require_data()
                self.snapshot()  # Invalidate results from a previous dataset before preparing.
                self.visual_review = parse_request(controller.dataset.current,data.get("question"))
                self.visual_batch = []
            elif action == "visual_prepare_batch":
                from .visual_requests import parse_requests
                frame=controller.require_data();self.snapshot()
                self.visual_batch=parse_requests(frame,data.get('question'))
                self.visual_review=self.visual_batch[0]
            elif action == "visual_select_review":
                self.snapshot()
                index=data.get('index')
                if type(index) is not int or not 0<=index<len(self.visual_batch):raise ValueError("Choose a question from the current list.")
                self.visual_review=self.visual_batch[index]
            elif action == "visual_execute":
                from .visual_requests import execute_request
                import uuid
                frame = controller.require_data()
                self.snapshot()
                result, figure = execute_request(frame,data.get("plan"))
                result["id"] = uuid.uuid4().hex
                result["question"] = str(data.get("question", "Custom visual plan"))[:4000]
                result["dataset"] = {"name":Path(controller.source).name,"version":controller.dataset.index}
                self.visual_figures[result["id"]] = figure
                try:self.render_visual_result(result)
                except Exception:
                    self.visual_figures.pop(result["id"],None)
                    raise
                self.visual_results.insert(0,result)
                for old in self.visual_results[6:]:self.visual_figures.pop(old["id"],None)
                self.visual_results = self.visual_results[:6]
                self.announce("Visual result ready. Review the findings and calculation details below.")
            elif action in {"visual_remove", "visual_export"}:
                self.snapshot()
                result = next((r for r in self.visual_results if r["id"]==data.get("id")),None)
                if result is None:raise ValueError("Choose a current visual result.")
                if action=="visual_remove":
                    self.visual_results.remove(result)
                    self.visual_figures.pop(result["id"],None)
                else:
                    format=data.get("format","csv")
                    if format not in {"csv","png","svg","html","json"}:raise ValueError("Choose a supported export format.")
                    path,_=QFileDialog.getSaveFileName(self.parent(),"Export visual result","visual-result."+format,format.upper()+" files (*."+format+")")
                    if path:
                        controller.protect_output(path)
                        table=result["table"]
                        if format=="csv":pd.DataFrame(table["data"],columns=table["columns"]).to_csv(path,index=False)
                        elif format in {"png","svg"}:
                            figure=self.visual_figures[result["id"]]
                            figure.savefig(path,format=format,dpi=180,bbox_inches="tight",facecolor=figure.get_facecolor())
                        elif format=="json":
                            payload={k:v for k,v in result.items() if k not in {"chart","id"}}
                            Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
                        else:
                            import html
                            body="<!doctype html><meta charset='utf-8'><title>ML Studio visual result</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#1d2946}table{border-collapse:collapse}td,th{border:1px solid #ccd3df;padding:8px}img{max-width:100%}</style>"
                            body+="<h1>"+html.escape(result["title"])+"</h1><p>"+html.escape(result["question"])+"</p><p>Dataset: "+html.escape(result["dataset"]["name"])+" · version "+str(result["dataset"]["version"])+"</p>"
                            body+="<ul>"+"".join("<li>"+html.escape(t)+"</li>" for t in result["insights"])+"</ul><p>"+html.escape(result["protocol"])+"</p>"
                            body+='<img alt="Calculated chart" src="'+result["chart"]+'">'+pd.DataFrame(table["data"],columns=table["columns"]).to_html(index=False,escape=True)
                            if result.get("support"):
                                support=result["support"]
                                body+="<h2>Group sizes and finite measurement counts</h2>"+pd.DataFrame(support["data"],columns=support["columns"]).to_html(index=False,escape=True)
                            body+="<h2>Calculation plan</h2><pre>"+html.escape(json.dumps(result["plan"],indent=2,ensure_ascii=False))+"</pre>"
                            Path(path).write_text(body,encoding="utf-8")
                        self.announce("Visual result exported.")
            elif action == "lab_analyze":
                from .lab import grouped_analysis
                result, figure = grouped_analysis(controller.require_data(), **{
                    k:v for k,v in data.items() if k in {"group", "columns", "statistic", "layout", "title", "xlabel", "ylabel"}})
                self.lab_result, self.lab_figure = result, figure
                self.lab_key = (id(controller.dataset), controller.dataset.index)
                self.render_lab()
                self.announce("Grouped summary and chart ready. All dataset rows were included.")
            elif action == "lab_export":
                if self.lab_result is None or self.lab_key != (id(controller.dataset), controller.dataset.index if controller.dataset else None):
                    raise ValueError("Create a grouped summary before exporting.")
                format = data.get("format", "csv")
                if format not in {"csv", "svg", "png"}:raise ValueError("Choose CSV, SVG or PNG.")
                path,_ = QFileDialog.getSaveFileName(self.parent(), "Export lab result", "grouped-summary."+format, format.upper()+" files (*."+format+")")
                if path:
                    controller.protect_output(path)
                    if format == "csv":
                        table = self.lab_result["table"]
                        pd.DataFrame(table["data"], columns=table["columns"]).to_csv(path, index=False)
                    else:self.lab_figure.savefig(path, format=format, dpi=180, bbox_inches="tight", facecolor=self.lab_figure.get_facecolor())
                    self.announce("Lab result exported.")
            elif action == "chart":
                for widget, key in [(controller.chart_type, "type"), (controller.chart_x, "x"), (controller.chart_y, "y")]:
                    widget.setCurrentText(data[key])
                controller.draw_chart()
                self.render_chart(update_key=True)
            elif action == "cluster_study":
                from .clustering import study
                frame = controller.require_data()
                key = (id(controller.dataset),controller.dataset.index)
                def complete(result):
                    self.clustering_result,self.clustering_key=result,key
                    controller.store.add(result["algorithm"]+" clustering",{"metrics":result["metrics"],"clustering":result})
                    controller.refresh_history()
                self.research._background(lambda:study(frame,**{k:v for k,v in data.items() if k in {"columns","algorithm","clusters","eps","min_samples","scale","compare"}}),"Groups are ready. Inspect the plot and group summaries.",complete)
            elif action == "cluster_export":
                if not self.clustering_result:
                    raise ValueError("Find groups before exporting labels.")
                path,_=QFileDialog.getSaveFileName(self.parent(),"Export grouped data","grouped-data.csv","CSV (*.csv)")
                if path:
                    controller.protect_output(path)
                    if Path(path).resolve()==Path(controller.source).resolve():raise ValueError("Choose a new file to preserve your source CSV.")
                    frame=controller.require_data()
                    name="cluster"
                    while name in frame:name="_"+name
                    frame.assign(**{name:self.clustering_result["labels"]}).to_csv(path,index=False)
                    self.announce("Grouped data exported. Label -1 means DBSCAN noise.")
            elif action == "cluster":
                self.comparison_start = None
                controller.clustering_algorithm.setCurrentText(data["algorithm"])
                controller.cluster_data()
                # The existing clustering implementation reports structured results.
                self.announce("Clustering results are recorded in Experiment History.")
            elif action == "workflow":
                nodes, edges = data.get("nodes"), data.get("edges")
                if not isinstance(nodes, dict) or not isinstance(edges, list) or len(nodes) > 100:
                    raise ValueError("Invalid workflow graph.")
                for key, node in nodes.items():
                    if not str(key).isdigit() or node.get("type") not in set(PORTS)-{'Fill missing'}:
                        raise ValueError("Unavailable workflow node.")
                    position = node.get("position")
                    if not isinstance(position, list) or len(position) != 2 or not all(isinstance(v, (float, int)) and abs(v) <= 10000 for v in position):
                        raise ValueError("Invalid node position.")
                for a, b in edges:
                    if a not in nodes or b not in nodes or a == b or PORTS[nodes[a]["type"]][1] != PORTS[nodes[b]["type"]][0]:
                        raise ValueError("Incompatible connection.")
                graph = Graph(nodes, edges)
                controller.canvas.restore(graph)
            elif action == "validate":
                controller.canvas.graph.order()
                self.announce("Workflow validated. Connections and execution order are valid.")
            else:
                raise ValueError("Unknown workspace command.")
            self.publish()
        except Exception as exc:
            logging.exception("Workspace command failed: %s", action)
            self.announce(str(exc), "error")
            self.publish()

    def render_dataset_visual(self):
        previous_figure,previous_chart=self.controller.figure,self.chart
        try:
            self.controller.figure=self.dataset_visual_figure
            self.render_chart()
            self.dataset_visual_result['chart']=self.chart
        finally:self.controller.figure,self.chart=previous_figure,previous_chart

    def render_visual_result(self, result):
        previous_figure, previous_chart = self.controller.figure, self.chart
        try:
            self.controller.figure = self.visual_figures[result["id"]]
            self.render_chart()
            result["chart"] = self.chart
        finally:
            self.controller.figure, self.chart = previous_figure, previous_chart

    def render_lab(self):
        previous_figure, previous_chart = self.controller.figure, self.chart
        try:
            self.controller.figure = self.lab_figure
            self.render_chart()
            self.lab_result["chart"] = self.chart
        finally:
            self.controller.figure, self.chart = previous_figure, previous_chart

    def render_chart(self,update_key=False):
        """Theme plotting paper and labels without inverting data colors."""
        from matplotlib.colors import to_rgba
        dark=self.controller.settings.value("referenceTheme","pastel")=="dark"
        paper,ink,edge=("#1a2133","#e9eef9","#46536e") if dark else ("#ffffff","#27334e","#c3ccdd")
        figure=self.controller.figure
        if figure._suptitle is not None:figure._suptitle.set_color(ink)
        figure.set_facecolor(paper)
        for axis in figure.axes:
            axis.set_facecolor(paper)
            axis.tick_params(colors=ink)
            for item in [axis.title,axis.xaxis.label,axis.yaxis.label,axis.xaxis.get_offset_text(),axis.yaxis.get_offset_text(),*axis.texts]:item.set_color(ink)
            for spine in axis.spines.values():spine.set_color(edge)
            if axis.get_legend() is not None:
                legend = axis.get_legend()
                legend.get_frame().set_facecolor(paper)
                legend.get_frame().set_edgecolor(edge)
                for text in legend.get_texts():text.set_color(ink)
            for line in axis.get_lines():
                if to_rgba(line.get_color()) in {to_rgba("black"),to_rgba("#e9eef9"),to_rgba("#27334e")}:
                    line.set_color(ink)
                    if line.get_markeredgecolor()!="auto":line.set_markeredgecolor(ink)
        output=io.BytesIO()
        figure.savefig(output,format="svg",bbox_inches="tight",facecolor=paper)
        self.chart="data:image/svg+xml;base64,"+base64.b64encode(output.getvalue()).decode()
        if update_key:self._chart_key=(id(self.controller.dataset),self.controller.dataset.index)


class ModernWindow(QMainWindow):
    def __init__(self, directory=None):
        super().__init__()
        self.setWindowTitle("ML Studio · Your offline ML workspace")
        self.resize(1460, 940)
        self.setMinimumSize(1080, 720)
        self.controller = MainWindow(directory)
        self.bridge = StudioBridge(self.controller, self)
        self.profile = QWebEngineProfile(self)  # off-the-record; no web cookies or disk cache
        self.interceptor = OfflineRequests(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view = QWebEngineView(self)
        self.web_page = OfflinePage(self.profile, self.view)
        theme=self.controller.settings.value("referenceTheme","pastel")
        if theme not in {"pastel","contrast","dark"}:theme="pastel"
        self.update_native_theme(theme)
        boot=QWebEngineScript()
        boot.setName("ML Studio appearance")
        boot.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        boot.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        boot.setRunsOnSubFrames(False)
        boot.setSourceCode("(()=>{const apply=()=>{if(!document.documentElement)return false;document.documentElement.dataset.theme="+json.dumps(theme)+";return true;};if(!apply()){const observer=new MutationObserver(()=>{if(apply())observer.disconnect();});observer.observe(document,{childList:true});}})();")
        self.web_page.scripts().insert(boot)
        self.view.setPage(self.web_page)
        self.channel = QWebChannel(self.web_page)
        self.channel.registerObject("studio", self.bridge)
        self.web_page.setWebChannel(self.channel)
        self.setCentralWidget(self.view)
        self.view.setUrl(QUrl.fromLocalFile(str(ASSETS / "index.html")))

    def update_native_theme(self,theme):
        paper,ink,border=("#1a2133","#edf2fc","#46536e") if theme=="dark" else ("#f7f9fc","#24324b","#cbd4e6")
        controls=("#232d43" if theme=="dark" else "#ffffff")
        rules=f"""
        QDialog {{ background: {paper}; }}
        QDialog QWidget {{ color: {ink}; }}
        QDialog QLabel {{ background: transparent; color: {ink}; }}
        QDialog QComboBox, QDialog QLineEdit, QDialog QTextEdit {{ background: {controls}; color: {ink}; border: 1px solid {border}; }}
        QComboBox QAbstractItemView {{ background: {controls}; color: {ink}; selection-background-color: #6740ca; selection-color: white; }}
        QDialog QPushButton {{ background: #6740ca; color: white; }}
        """
        QApplication.instance().setStyleSheet(THEME+rules)

    def closeEvent(self, event):
        if self.controller.busy:
            QMessageBox.information(self, "Operation running", "Wait for the current operation to finish before closing.")
            event.ignore()
            return
        self.controller.settings.setValue("modernGeometry", self.saveGeometry())
        self.controller.close()
        event.accept()
