"""A concise, escaped result brief built only from the fitted run's own evidence."""
import html
import json
import math


def result_brief(result):
    if result is None:raise ValueError('Train or open a fitted model before creating a result brief.')
    metadata=result.metadata;diagnostic=metadata.get('diagnostics',{});baseline=diagnostic.get('baseline',{})
    metric=baseline.get('metric') or ('RMSE' if 'RMSE' in result.metrics else 'Accuracy')
    def number(value):return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None
    score=number(result.metrics.get(metric));reference=number(baseline.get('value'))
    improvement=None if score is None or reference is None else reference-score if metric=='RMSE' else score-reference
    return {'algorithm':metadata.get('algorithm'),'target':metadata.get('target'),'metric':metric,'score':score,
        'baseline':reference,'baseline_name':baseline.get('name'),'improvement':improvement,
        'training_rows':metadata.get('training_rows'),'test_rows':metadata.get('test_rows'),'input_count':len(metadata.get('features',[])),
        'protocol':metadata.get('protocol'),'fingerprint':metadata.get('dataset_fingerprint'),
        'preparation':metadata.get('preparation',{}),'metrics':result.metrics,'warnings':metadata.get('warnings',[]),
        'interpretation':'A positive improvement beats the simple baseline on this same holdout. Repeated choices after viewing it need new independent final-test data.'}


def brief_html(result):
    brief=result_brief(result);esc=lambda value:html.escape(str(value))
    facts=[('Method',brief['algorithm']),('Outcome',brief['target']),('Protocol',brief['protocol']),
        ('Training records',brief['training_rows']),('Held-out records',brief['test_rows']),('Input columns',brief['input_count']),
        (brief['metric'],brief['score']),('Baseline',brief['baseline']),('Improvement over baseline',brief['improvement'])]
    return '<!doctype html><html><head><meta charset="utf-8"><title>ML Studio · Result brief</title><style>body{font:15px system-ui;color:#26334a;max-width:900px;margin:48px auto;padding:0 24px}h1{font-size:28px}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #d7dfeb}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5fa;padding:18px}li{margin:8px 0}</style></head><body><h1>ML Studio · Result brief</h1><p>Evidence from the fitted model run. Current workspace contents are not substituted for its training dataset.</p><table>'+''.join('<tr><th>'+esc(k)+'</th><td>'+esc(v if v is not None else 'Not available')+'</td></tr>' for k,v in facts)+'</table><p>'+esc(brief['interpretation'])+'</p><h2>Checks to explain</h2><ul>'+''.join('<li>'+esc(w)+'</li>' for w in brief['warnings'])+'</ul><h2>Metrics</h2><pre>'+esc(json.dumps(brief['metrics'],indent=2,ensure_ascii=False))+'</pre><h2>Training preparation</h2><pre>'+esc(json.dumps(brief['preparation'],indent=2,ensure_ascii=False))+'</pre><p>Training dataset fingerprint: '+esc(brief['fingerprint'])+'</p></body></html>'
