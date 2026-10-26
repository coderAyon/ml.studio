"""Local, self-contained source/data research bundles and deterministic replay."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
from . import __version__
from .research import ResearchConfig, ResearchSession


def export_pack(path,session,provenance):
    destination=Path(path)
    if destination.suffix.lower()!=".zip":raise ValueError("Choose a .zip filename for the research pack.")
    dataset=session.frame.to_csv(index=False).encode("utf-8")
    typed=session.frame.to_json(orient="table",index=False,date_format="iso",double_precision=15).encode("utf-8")
    manifest={"schema":1,"software_version":__version__,"python":platform.python_version(),"versions":{p:version(p) for p in ["numpy","pandas","scipy","scikit-learn"]},"dataset_sha256":hashlib.sha256(dataset).hexdigest(),"typed_data_sha256":hashlib.sha256(typed).hexdigest(),"provenance":provenance,"config":asdict(session.config),"runs":session.run_log,"contains_data":True}
    text=lambda value:json.dumps(value,indent=2,default=str,allow_nan=False).encode("utf-8")
    contents={"dataset.csv":dataset,"frame.json":typed,"manifest.json":text(manifest),"splits.json":text(session.manifest()),"results.json":text(session.snapshot()),"requirements.txt":"\n".join(f"{k}=={v}" for k,v in manifest["versions"].items()).encode(),"reproduce.py":b'from pathlib import Path\nfrom mlstudio.research_export import replay_pack\nif __name__ == "__main__":\n    replay_pack(Path(__file__).resolve().parent)\n',"README.txt":b'ML Studio Research Pack\nContains the active dataset, transformed values, positional split definitions and original-source provenance. Treat it as private data. No model pickle is included.\n\nInstall the exact requirements into a compatible Python environment, then run: python reproduce.py\nThe replay checks both dataset fingerprints and split definitions, reruns the recorded experiments and compares metrics with tolerance 1e-9. Runtime timing is not expected to match. Final-test errors are descriptive; model choices informed by them require new external data.\n\nSources are included; this is a generated Python project. Inspect scripts before executing research packs obtained from other people.\n'}
    sources=Path(getattr(sys,"_MEIPASS",Path(__file__).parent)) / "replay_sources" if getattr(sys,"frozen",False) else Path(__file__).parent
    for name in ["__init__.py","training.py","training_prep.py","model_settings.py","evaluation.py","research.py","safety.py","error_analysis.py","research_export.py"]:
        contents["mlstudio/"+name]=(sources/name).read_bytes()
    from uuid import uuid4
    temporary=destination.with_name(destination.name+"."+uuid4().hex+".tmp")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in contents.items():archive.writestr(name,payload)
    temporary.replace(destination)


def replay_pack(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    for filename,key in [("dataset.csv","dataset_sha256"),("frame.json","typed_data_sha256")]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=manifest[key]:raise ValueError(f"Dataset fingerprint mismatch: {filename}")
    for package,expected in manifest["versions"].items():
        if version(package)!=expected:raise ValueError(f"Version mismatch for {package}; install {expected} for reproducible replay.")
    frame=pd.read_json(directory/"frame.json",orient="table")
    session=ResearchSession(frame,ResearchConfig(**manifest["config"]),split_manifest=json.loads((directory/"splits.json").read_text()))
    expected=json.loads((directory/"results.json").read_text())
    for run in manifest["runs"]:
        if run["kind"]=="comparison":session.compare(run["algorithms"])
        elif run["kind"]=="impact":session.impact(run["algorithm"])
        elif run["kind"]=="final":session.evaluate_final(run["algorithm"],run["variant"])
        else:raise ValueError("Unknown recorded experiment kind.")
    observed=session.snapshot()
    for section,key in [("comparison","models"),("impact","variants")]:
        if expected[section] is None:continue
        lookup={(r["algorithm"],r["variant"]):r for r in observed[section][key]}
        for row in expected[section][key]:
            actual=lookup[(row["algorithm"],row["variant"])]
            if not np.allclose(row["scores"],actual["scores"],rtol=1e-9,atol=1e-9):raise ValueError(f"Replayed {section} scores differ from the original.")
    if expected["final"]:
        for metric,value in expected["final"]["metrics"].items():
            actual=observed["final"]["metrics"][metric]
            if value is None and actual is None:continue
            if not np.isclose(value,actual,rtol=1e-9,atol=1e-9):raise ValueError(f"Replayed final metric differs: {metric}")
    output={"verified":True,"config":asdict(session.config),"comparison":session.comparison,"impact":session.impact_result,"final":session.final_result}
    (directory/"reproduced.json").write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")
    print("Verified: fingerprints, exact splits and recorded metrics reproduced.")
    return output
