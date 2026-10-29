"""Business mode coordination, revision-safe results and local report exports."""
import html
import json
from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import QFileDialog

from .business import analyze_business


class BusinessWorkspace:
    def __init__(self,bridge):
        self.bridge=bridge;self.result=None;self.key=None;self.positions=[]

    def revision(self):
        c=self.bridge.controller
        return c.dataset.revision_id if c.dataset else None

    def state(self):
        c=self.bridge.controller
        return {'enabled':c.settings.value('businessMode',False,type=bool),
                'config':getattr(c,'business_config',{}),
                'result':self.result if self.key==self.revision() else None}

    def command(self,action,data):
        c=self.bridge.controller
        if action=='business_mode':
            if type(data.get('enabled')) is not bool:raise ValueError('Choose Business mode on or off.')
            c.settings.setValue('businessMode',data['enabled']);return
        if action=='business_run':
            frame=c.require_data();key=self.revision();config=data.get('config',{})
            def complete(report):
                if key!=self.revision():return
                self.positions=report.pop('_positions');self.result=report;self.key=key
                c.business_config=report['config']
            self.bridge.research._background(lambda:analyze_business(frame,config),'Business report calculated from the current filtered records.',complete)
            return
        if action=='business_definition_open':
            path,_=QFileDialog.getOpenFileName(self.bridge.parent(),'Open business report definition','','ML Studio business (*.json)')
            if not path:return
            p=Path(path)
            if p.stat().st_size>100000:raise ValueError('Report definition is too large.')
            payload=json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(payload,dict) or type(payload.get('schema')) is not int or payload['schema']!=1 or not isinstance(payload.get('config'),dict):
                raise ValueError('Unsupported business report definition.')
            # Validate against actual active data before changing saved configuration.
            return self.command('business_run',{'config':payload['config']})
        if action not in {'business_export','business_definition_save'}:raise ValueError('Unknown business action.')
        if self.key!=self.revision() or self.result is None:raise ValueError('Build a business report for the current dataset before exporting.')
        result=self.result
        if action=='business_definition_save':
            path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Save report definition','business-report.json','JSON (*.json)')
            if path:
                c.protect_output(path)
                _atomic(path,json.dumps({'schema':1,'config':result['config']},indent=2,allow_nan=False))
            return
        kind=data.get('kind','html')
        if kind not in {'html','summary','segments','trend','pivot','data'}:raise ValueError('Choose an available business export.')
        path,_=QFileDialog.getSaveFileName(self.bridge.parent(),'Export business report','business-report.'+('html' if kind=='html' else 'csv'),
                                        'HTML (*.html)' if kind=='html' else 'CSV (*.csv)')
        if not path:return
        c.protect_output(path)
        if kind=='data':
            _atomic(path,c.require_data().iloc[self.positions].to_csv(index=False));return
        if kind=='summary':
            table=pd.DataFrame([{'metric':k,'value':v} for k,v in {'records':result['filtered_rows'],
                'selected_metric':result['metric']['value'],'distinct_customers':result['distinct_customers'],
                'target_gap':result['target']['gap'],'what_if':result['scenario']['value']}.items()])
        else:
            part=result.get(kind if kind!='trend' else 'trend') if kind!='html' else None
            if kind!='html':
                if part is None:raise ValueError('Configure this analysis before exporting it.')
                part=part.get('table',part);table=pd.DataFrame(part['data'],columns=part['columns'])
        if kind!='html':_atomic(path,table.to_csv(index=False));return
        sections=[]
        for key in ['segments','trend','pivot','detail']:
            part=result.get(key)
            if part:
                part=part.get('table',part)
                sections.append('<h2>'+html.escape(key.title())+'</h2>'+pd.DataFrame(part['data'],columns=part['columns']).to_html(index=False,escape=True))
        text='<!doctype html><html><head><meta charset="utf-8"><title>Business report</title><style>body{font:15px system-ui;margin:40px;color:#292039;max-width:1200px}h1,h2{color:#603bb5}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}th{background:#eee6fa}pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#5d536d}</style></head><body>'
        text+='<h1>'+html.escape(result['config']['title'])+'</h1><p>'+str(result['filtered_rows'])+' filtered records / '+str(result['total_rows'])+' total</p>'
        text+='<h2>Selected metric</h2><p>'+html.escape(str(result['metric']['value']))+'</p>'
        text+='<h2>Calculation settings and evidence</h2><pre>'+html.escape(json.dumps({k:result[k] for k in ['config','metric','finance','target','scenario','quality','notes']},indent=2,allow_nan=False))+'</pre>'
        text+=''.join(sections)+'<small>Local descriptive report; contains dataset values. Drill-down preview is bounded. Export filtered data separately for all rows/columns.</small></body></html>'
        _atomic(path,text)


def _atomic(path,text):
    from uuid import uuid4
    dest=Path(path);temp=dest.with_name(dest.name+'.'+uuid4().hex+'.tmp')
    try:temp.write_text(text,encoding='utf-8',newline='');temp.replace(dest)
    finally:temp.unlink(missing_ok=True)
