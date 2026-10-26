"""Escaped, self-contained HTML exports."""
from dataclasses import asdict
from datetime import datetime
from html import escape
import json
from pathlib import Path
from .analyzer import profile
from . import __version__


def export_report(path, frame, issues, transformations, results):
    payload = {"version": __version__, "generated": datetime.now().astimezone().isoformat(), "dataset": profile(frame), "findings": [asdict(i) for i in issues], "transformations": transformations, "results": results}
    body = escape(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>ML Studio report</title><style>body{max-width:1000px;margin:48px auto;font:16px Segoe UI;color:#24324b}pre{white-space:pre-wrap;background:#f4f6fa;padding:24px;border-radius:12px}</style><h1>ML Studio · Experiment report</h1><p>Local analysis. Potential findings require domain review. Statistical association does not establish causation.</p><pre>'+body+'</pre>', encoding="utf-8")
