"""Qt coordination for local, reviewable prediction and model evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog
from .data import load_csv
from .prediction import prediction_table


class PredictionWorkspace:
    def __init__(self,bridge):
        self.bridge=bridge;self.key=None;self.clear()

    def clear(self):
        self.input=None;self.path=None;self.output=None;self.summary=None;self.single=None;self.explanation=None

    def state(self):
        result=self.bridge.controller.result
        if self.key is not result:self.key=result;self.clear()
        return {'input':None if self.input is None else {'name':Path(self.path).name,'rows':len(self.input),'columns':list(self.input.columns),
                    'preview':json.loads(self.input.iloc[:8,:12].to_json(orient='split',date_format='iso'))},
                'output':self.summary,'preview':None if self.output is None else self.preview(),
                'single':self.single,'explanation':self.explanation}

    def preview(self):
        summary=self.summary;columns=list(self.output.columns[:12])
        columns=list(dict.fromkeys(columns+[summary['prediction_column']]+summary['probability_columns'][:20]))
        return json.loads(self.output.loc[:,columns].head(50).to_json(orient='split',date_format='iso'))

    def command(self,action,data):
        c=self.bridge.controller;self.state();result=c.result
        if result is None:raise ValueError('Train or open a saved model first.')
        if action=='prediction_import':
            options=self.bridge.prediction_options(data)
            path,_=QFileDialog.getOpenFileName(self.bridge.parent(),'Choose prediction CSV','','CSV and text (*.csv *.tsv *.txt)')
            if not path:return
            def read():
                frame=load_csv(path,**options)
                if len(frame)>20000 or len(frame.columns)>300:raise ValueError('Prediction preview supports up to 20,000 rows and 300 CSV columns.')
                return frame
            def done(frame):self.input=frame;self.path=path;self.output=self.summary=None
            self.bridge.research._background(read,'Prediction CSV loaded. Review its columns before running.',done)
        elif action=='prediction_run':
            if self.input is None:raise ValueError('Choose a prediction CSV first.')
            if type(data.get('allow_extra',False)) is not bool:raise ValueError('Choose whether extra CSV columns may be kept.')
            source=self.input.copy(deep=True)
            def done(value):self.output,self.summary=value
            self.bridge.research._background(lambda:prediction_table(result,source,data.get('mapping'),data.get('allow_extra',False)),
                'Predictions ready. Inspect the preview before exporting.',done)
        elif action=='prediction_export':
            if self.output is None:raise ValueError('Generate predictions first.')
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Save predictions','predictions.csv','CSV (*.csv)')
            if path:
                c.protect_output(path)
                if self.path and Path(path).resolve()==Path(self.path).resolve():raise ValueError('Choose a separate output to preserve the prediction input.')
                self.output.to_csv(path,index=False);self.bridge.announce('Prediction table exported, including available class probabilities.')
        elif action=='prediction_single':
            values=data.get('values',{})
            if not isinstance(values,dict) or set(values)!=set(result.metadata['features']):raise ValueError('Supply each expected input field.')
            if any(not isinstance(v,(str,int,float,bool,type(None))) for v in values.values()):raise ValueError('Use one value per input.')
            frame=pd.DataFrame([{c:(np.nan if v is None or v=='' else v) for c,v in values.items()}])
            out,summary=prediction_table(result,frame)
            self.single={'table':json.loads(out.to_json(orient='split')),'summary':summary}
        elif action=='prediction_explain':
            from .evaluation import explain_inputs
            frame=c.require_data()
            def done(value):self.explanation=value
            self.bridge.research._background(lambda:explain_inputs(result,frame),'Held-out input explanation calculated.',done)
        elif action=='prediction_diagnostics_export':
            diagnostic=result.metadata.get('diagnostics')
            if not diagnostic:raise ValueError('This model has no holdout records. Train a new experiment.')
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Export shown holdout records','holdout-records.csv','CSV (*.csv)')
            if path:
                c.protect_output(path)
                rows=[{**{k:v for k,v in r.items() if k!='features'},**{'input: '+c:v for c,v in r['features'].items()}} for r in diagnostic['records']]
                pd.DataFrame(rows).to_csv(path,index=False)
        else:raise ValueError('Unknown prediction action.')
