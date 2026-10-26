"""Qt coordination for research experiments; CPU work stays off the UI thread."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog
from .research import ResearchConfig, ResearchSession
from .research_export import export_pack
from .safety import safety_advice
from .training import REGRESSION


class ResearchCoordinator(QObject):
    progress=Signal(str)

    def __init__(self,bridge):
        super().__init__(bridge)
        self.bridge=bridge
        self.options={"protocol":"random","group":"","time_column":"","excluded":[],"folds":3,"repeats":2}
        self.session=None
        self.key=None
        self.cached=None
        self.prior_exposures=0
        self._safety_key=None
        self._safety=[]
        self.progress.connect(bridge.on_status)

    def config(self):
        controller=self.bridge.controller
        return ResearchConfig(target=controller.target.currentText(),task="regression" if controller.algorithm.currentText() in REGRESSION else "classification",**self.options)

    def signature(self):
        controller=self.bridge.controller
        return (id(controller.dataset),controller.dataset.index if controller.dataset else None,json.dumps(asdict(self.config()),sort_keys=True))

    def invalidate_if_changed(self):
        if self.key is not None and self.key!=self.signature():
            self.session,self.cached,self.key=None,None,None

    def state(self):
        self.invalidate_if_changed()
        controller=self.bridge.controller
        cfg=self.config()
        sig=self.signature()
        if controller.dataset and sig!=self._safety_key:
            self._safety=safety_advice(controller.dataset.current,cfg)
            self._safety_key=sig
        return {"options":self.options,"config":asdict(cfg),"safety":self._safety if controller.dataset else [],"session":self.cached,"prior_final_exposures":self.prior_exposures,"final_locked":bool(self.session and self.session.final_used),"data_in_pack":"Research ZIP contains the active dataset and original-source path. It is saved locally; no data is uploaded."}

    def command(self,action,data):
        controller=self.bridge.controller
        self.invalidate_if_changed()
        if action=="research_config":
            if self.session and self.session.final_used and not data.get("acknowledge_reset"):
                raise ValueError("Final outcomes were inspected. Resetting cannot make the same data independent; explicitly acknowledge a new exploratory study.")
            proposed={key:data.get(key,value) for key,value in self.options.items()}
            cfg=ResearchConfig(target=controller.target.currentText(),task="regression" if controller.algorithm.currentText() in REGRESSION else "classification",**proposed)
            if cfg.protocol not in {"random","group","time"} or not isinstance(cfg.excluded,list) or not all(isinstance(c,str) for c in cfg.excluded):raise ValueError("Invalid research settings.")
            if not isinstance(cfg.folds,int) or not isinstance(cfg.repeats,int) or not 2<=cfg.folds<=5 or not 1<=cfg.repeats<=3:raise ValueError("Use 2–5 folds, 1–3 repeats.")
            if not isinstance(cfg.group,str) or not isinstance(cfg.time_column,str):raise ValueError("Group/time column names must be strings.")
            self.options=proposed
            self.session,self.cached,self.key=None,None,None
            self.bridge.announce("Research design saved. Previous results cleared; no data was changed.")
            return
        if action=="research_export":
            if self.session is None:raise ValueError("Run a research experiment before exporting a research pack.")
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),"Export research pack (includes active dataset)","MLStudio-Research.zip","ZIP (*.zip)")
            if not path:return
            controller.protect_output(path)
            source=Path(controller.source)
            if Path(path).resolve()==source.resolve():raise ValueError("Choose a new output path.")
            provenance={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,"active_version":controller.dataset.index,"transformations":controller.dataset.log[:controller.dataset.index],"prior_final_exposures":self.prior_exposures,"offline":True}
            session=self.session
            self._background(lambda:export_pack(path,session,provenance),"Research pack exported locally, including active data and runnable Python.",persist=False)
            return
        if action not in {"research_compare","research_impact","research_final"}:raise ValueError("Unknown research command.")
        frame=controller.require_data()
        budget=controller.settings.value("researchRowBudget",20000,type=int)
        if len(frame)>budget:
            raise ValueError(f"Dataset exceeds the configured research budget of {budget:,} rows. Adjust it in Settings or use a smaller extract.")
        from .training_prep import assert_evaluation_ready
        assert_evaluation_ready(controller.dataset)
        cfg=self.config();sig=self.signature()
        if not cfg.target:raise ValueError("Select the outcome column in Safety Advisor first.")
        algorithms=data.get("algorithms",[])
        algorithm=data.get("algorithm",controller.algorithm.currentText())
        variant=data.get("variant","baseline")
        if self.session is None:
            # Source-row content fingerprint also audits target/protocol changes across sessions.
            source=Path(controller.source)
            fingerprint=hashlib.sha256(source.read_bytes() if source.is_file() else frame.to_json(orient="split",date_format="iso").encode()).hexdigest()
        else:fingerprint=self.fingerprint
        if action=="research_final":
            if not data.get("confirmed"):raise ValueError("Confirm your candidate before unlocking the reserved final test.")
            if self.session is None or (self.session.comparison is None and self.session.impact_result is None):raise ValueError("Run development CV or an impact experiment before final evaluation.")
            if self.session.final_used:raise ValueError("Final test already inspected. Error Detective shows that one evaluation.")
            if self.prior_exposures and not data.get("acknowledge_exposure"):raise ValueError("This dataset has previous final-test exposure. Acknowledge that this result is exploratory, or use new external data.")
            controller.store.record_final_exposure(fingerprint,hashlib.sha256(json.dumps(self.session.test_positions).encode()).hexdigest())
        def work():
            session=self.session or ResearchSession(frame,cfg)
            if action=="research_compare":session.compare(algorithms,self.progress.emit)
            elif action=="research_impact":session.impact(algorithm,self.progress.emit)
            else:session.evaluate_final(algorithm,variant)
            return session
        def complete(session):
            self.session=session;self.key=sig;self.fingerprint=fingerprint
            self.cached=session.snapshot()
            self.prior_exposures=controller.store.final_exposure_count(fingerprint)-(1 if action=="research_final" else 0)
            payload={"metadata":{"target":cfg.target,"protocol":cfg.protocol,"features":session.features,"research_config":asdict(cfg)},"metrics":session.final_result["metrics"] if session.final_result else {},"research":self.cached}
            controller.store.add({"research_compare":"Reliability CV","research_impact":"Preprocessing Impact","research_final":"Final Evaluation"}[action],payload)
            controller.refresh_history()
            self.bridge.publish()
        self._background(work,"Research experiment completed. Results are stored locally.",complete)

    def _background(self,function,message,complete=None,persist=True):
        # Convert worker errors to the embedded notification, keeping hidden native
        # controller message boxes from trapping the modern interface.
        def guarded():
            try:return {"value":function()}
            except Exception as exc:return {"error":str(exc)}
        def done(result):
            if "error" in result:
                if self.session:self.cached=self.session.snapshot()
                self.bridge.announce(result["error"],"error")
            else:
                if complete:complete(result["value"])
                self.bridge.announce(message)
            self.bridge.publish()
        self.bridge.controller.background(guarded,done,"Running local research experiment…")
