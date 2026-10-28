"""Small bridge for association mining, holdout diagnostics and ARFF exchange."""
import pandas as pd
from PySide6.QtWidgets import QFileDialog


class ComparisonTools:
    def __init__(self, bridge):
        self.bridge=bridge
        self.rules=None; self.rules_key=None
        self.diagnostics=None; self.diagnostics_key=None

    def key(self):
        c=self.bridge.controller
        return (id(c.dataset), c.dataset.index) if c.dataset else None

    def model_key(self):
        return (self.key(), id(self.bridge.controller.result))

    def state(self):
        return {'association': {'result':self.rules if self.rules_key==self.key() else None},
                'model_diagnostics':self.diagnostics if self.diagnostics_key==self.model_key() else None}

    def command(self, action, data):
        c=self.bridge.controller
        if action=='association_run':
            from .association import mine_rules
            frame=c.require_data().copy(deep=True);key=self.key()
            self.rules=None
            def complete(report):
                if key==self.key(): self.rules=report;self.rules_key=key
            self.bridge.research._background(lambda:mine_rules(frame,data),'Association rules are ready.',complete)
        elif action=='model_diagnostics_build':
            from .model_diagnostics import build_diagnostics
            frame=c.require_data().copy(deep=True);result=c.result;key=self.model_key()
            self.diagnostics=None
            def complete(report):
                if key==self.model_key(): self.diagnostics=report;self.diagnostics_key=key
            self.bridge.research._background(lambda:build_diagnostics(result,frame,data.get('positive_class')),
                'Held-out diagnostics are ready.',complete)
        elif action in {'association_export','model_diagnostics_export'}:
            if action=='association_export':
                if self.rules_key!=self.key() or self.rules is None:raise ValueError('Find current association rules before exporting.')
                table=self.rules['table'];frame=pd.DataFrame(table['data'],columns=table['columns']);name='association-rules.csv'
            else:
                from .model_diagnostics import diagnostics_rows
                if self.diagnostics_key!=self.model_key() or self.diagnostics is None:raise ValueError('Build current model diagnostics before exporting.')
                frame=diagnostics_rows(self.diagnostics,data.get('table','predictions'));name='model-diagnostics.csv'
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Save results',name,'CSV (*.csv)')
            if path:c.protect_output(path);frame.to_csv(path,index=False);self.bridge.announce('Results exported.')
        elif action=='export_arff':
            from .arff_io import export_arff
            frame=c.require_data()
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Save Weka dataset','dataset.arff','Weka ARFF (*.arff)')
            if path:c.protect_output(path);export_arff(frame,path);self.bridge.announce('Weka ARFF exported.')
        elif action=='model_diagnostics_image':
            import base64
            if self.diagnostics_key!=self.model_key() or self.diagnostics is None:raise ValueError('Build current diagnostics first.')
            chart=next((v for v in self.diagnostics['charts'] if v['id']==data.get('id')),None)
            if chart is None:raise ValueError('Choose an available chart.')
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Save diagnostic chart',chart['id']+'.svg','SVG (*.svg)')
            if path:
                from pathlib import Path
                c.protect_output(path);Path(path).write_bytes(base64.b64decode(chart['image'].split(',',1)[1]));self.bridge.announce('Chart saved.')
        else:raise ValueError('Unknown analysis tool action.')
