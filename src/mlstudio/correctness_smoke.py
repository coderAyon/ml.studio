"""Native correctness acceptance helpers; only labeled temporary QA fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QFileDialog

from .data import DatasetVersions, load_csv
from .ui_smoke import evaluate, wait_for
from .workflow import Graph


def _wait_export(window, path):
    loop, timer = QEventLoop(), QTimer()
    timer.timeout.connect(lambda: loop.quit() if path.is_file() and not window.controller.busy else None)
    timer.start(20)
    QTimer.singleShot(60000, loop.quit)
    loop.exec()
    timer.stop()
    assert path.is_file() and not window.controller.busy, f'Export did not finish: {path.name}'


def persistence(window, out):
    c, b = window.controller, window.bridge
    js = lambda text: evaluate(window, text)
    source = out / 'unsigned-boundary.csv'
    source.write_text('identifier,label\n0,A\n18446744073709551615,B\n', encoding='utf-8')
    original = source.read_bytes()
    c.canvas.restore(Graph())
    c.source = str(source)
    c.dataset = DatasetVersions(load_csv(source))
    c.import_options, c.preparation = {}, None
    c.refresh_data()
    b.publish()
    project = out / 'unsigned-project.json'
    QFileDialog.getSaveFileName = lambda *args: (str(project), '')
    js("navigate('data');cmd('save');true")
    assert project.is_file() and source.read_bytes() == original
    # Reopen uses the embedded frame even if the source path is unavailable.
    moved = source.with_name('unsigned-source-preserved.csv')
    source.replace(moved)
    QFileDialog.getOpenFileName = lambda *args: (str(project), '')
    previous_revision = c.dataset.revision_id
    js("cmd('open');true")
    wait_for(window, f"!state.busy&&state.dataset?.revision_id!=={json.dumps(previous_revision)}&&state.dataset?.rows===2")
    assert str(c.dataset.current.identifier.dtype) == 'uint64'
    assert str(c.dataset.current.identifier.iloc[1]) == '18446744073709551615'
    csv = out / 'unsigned-export.csv'
    QFileDialog.getSaveFileName = lambda *args: (str(csv), '')
    js("cmd('export_csv');true")
    assert '18446744073709551615' in csv.read_text() and moved.read_bytes() == original

    nominal = out / 'nominal-source.arff'
    nominal.write_text("@RELATION audit\n@ATTRIBUTE group {A,B,unused}\n@ATTRIBUTE x NUMERIC\n@DATA\nA,1\nB,2\n", encoding='utf-8')
    frame = load_csv(nominal)
    c.source, c.dataset = str(nominal), DatasetVersions(frame)
    c.refresh_data()
    b.publish()
    nominal_project = out / 'nominal-project.json'
    QFileDialog.getSaveFileName = lambda *args: (str(nominal_project), '')
    js("cmd('save');true")
    QFileDialog.getOpenFileName = lambda *args: (str(nominal_project), '')
    previous_revision = c.dataset.revision_id
    js("cmd('open');true")
    wait_for(window, f"!state.busy&&state.dataset?.revision_id!=={json.dumps(previous_revision)}&&state.dataset?.name==='nominal-source.arff'")
    pd.testing.assert_frame_equal(c.dataset.current, frame)
    assert c.dataset.current.attrs == frame.attrs

    # Exercise the normal Research bridge and native export dialog with precise time.
    frame = pd.DataFrame({'time': pd.date_range('2026-01-01', periods=100, freq='ns', tz='Asia/Dhaka'),
                          'x': np.arange(100, dtype=float), 'label': [0, 1] * 50})
    time_source = out / 'time-source.csv'
    frame.to_csv(time_source, index=False)
    c.source, c.dataset, c.preparation = str(time_source), DatasetVersions(frame), None
    c.refresh_data()
    c.target.setCurrentText('label')
    c.algorithm.setCurrentText('Decision Tree')
    b.publish()
    js("navigate('reliability');cmd('research_config',{protocol:'time',time_column:'time',folds:2,repeats:1});true")
    js("cmd('research_compare',{algorithms:['Decision Tree']});true")
    wait_for(window, "!!state.research.session?.comparison&&!state.busy", timeout=60)
    pack = out / 'nanosecond-study.zip'
    QFileDialog.getSaveFileName = lambda *args: (str(pack), '')
    js("cmd('research_export');true")
    _wait_export(window, pack)
    with zipfile.ZipFile(pack) as archive:
        assert json.loads(archive.read('manifest.json'))['schema'] == 2
        directory = out / 'nanosecond-replay'
        archive.extractall(directory)  # Our own generated archive, never external input.
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    # Source launcher smoke uses pythonw; find the explicit interpreter for captured replay output.
    interpreter = Path(sys.executable).with_name('python.exe')
    script = """from pathlib import Path
from mlstudio.research_export import replay_pack
replay_pack(Path.cwd())
import sys
assert all(Path(m.__file__).resolve().is_relative_to(Path.cwd().resolve()) for n,m in sys.modules.items()
           if n=='mlstudio' or n.startswith('mlstudio.'))
"""
    process = subprocess.run([str(interpreter), '-c', script], cwd=directory, env=environment,
                             capture_output=True, text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((directory / 'reproduced.json').read_text())['verified']
    return ['Unsigned project reopen/export preserves exact boundary values without source',
            'Nominal ARFF project preserves unused categories and provenance',
            'Native research export and isolated replay preserve nanosecond timezone splits']


def evidence_contracts(window, out):
    """Exercise repaired contracts through actual rendered controls and Qt jobs."""
    from .analyzer import analyze
    from .arff_io import export_arff
    c, b = window.controller, window.bridge
    js = lambda text: evaluate(window, text)
    notices = []
    b.notice.connect(lambda message: notices.append(json.loads(message)))
    frame = pd.DataFrame({'x': np.arange(80, dtype=float),
        'city': pd.Categorical([' A', 'b', None, 'A'] * 20, categories=[' A','b','A','unused']),
        'label': [0, 1] * 40})
    source = out / 'synthetic-nominal-cleaning.arff'; export_arff(frame, source)
    original = source.read_bytes()
    c.canvas.restore(Graph()); c.source = str(source); c.import_options = {}
    c.dataset = DatasetVersions(frame); c.issues = analyze(frame); c.refresh_data(); b.publish()
    issue = next(i for i in c.issues if i.title == 'Equivalent category labels')
    js(f"navigate('preprocessing');document.querySelector('[data-prep-issue=\"{issue.id}\"]').click();true")
    wait_for(window, 'state.preprocessing.detail?.affected_count===40')
    assert issue.count == b.preprocess_detail['affected_count'] == 40
    js("document.querySelector('[data-prep-operation=\"Trim whitespace\"]').click();true")
    wait_for(window, 'state.preprocessing.preview?.summary.changed_cells===20')
    assert c.dataset.current.city.iloc[0] == ' A'
    js("document.querySelector('[data-action=preprocess_apply]').click();true")
    wait_for(window, 'state.dataset?.version===1&&!state.busy')
    assert c.dataset.current.city.iloc[0] == 'A'
    js("cmd('undo');true"); wait_for(window, 'state.dataset.version===0&&!state.busy')
    assert c.dataset.current.city.iloc[0] == ' A'
    js("cmd('redo');true"); wait_for(window, 'state.dataset.version===1&&!state.busy')
    assert c.dataset.current.city.iloc[0] == 'A' and source.read_bytes() == original
    js("cmd('simple_train',{target:'label',task:'classification',algorithm:'Decision Tree'});true")
    wait_for(window, "state.model?.metadata.algorithm==='Decision Tree'&&!state.busy", timeout=60)
    js("cmd('prediction_explain');true")
    wait_for(window, '!!state.predictions.explanation&&!state.busy', timeout=60)
    assert b.predictions.explanation['sample_rows'] == 20
    model = out / 'validated-model.mlmodel'
    QFileDialog.getSaveFileName = lambda *args: (str(model), '')
    js("cmd('save_model');true"); assert model.is_file()
    # Model-only prediction must not depend on keeping the original data open.
    c.result = None; c.dataset = None; c.source = None; b.publish()
    QFileDialog.getOpenFileName = lambda *args: (str(model), '')
    js("cmd('load_model');true")
    wait_for(window, '!!state.model&&!state.busy', timeout=60)
    js("cmd('prediction_single',{values:{x:1,city:'A'}});true")
    wait_for(window, '!!state.predictions.single&&!state.busy')
    assert c.dataset is None and len(b.predictions.single['table']['data']) == 1
    # A renamed dataset cannot leave the previous explanation on screen.
    c.dataset = DatasetVersions(frame); c.source = str(source); c.issues = analyze(frame)
    c.refresh_data(); b.publish()
    count = len(notices)
    js("cmd('prediction_explain');true")
    wait_for(window, '!state.busy', timeout=60)
    assert any(n['kind'] == 'error' for n in notices[count:])
    assert b.predictions.explanation is None
    # Missing-version manifests must fail without replacing the active safe model.
    bad = out / 'invalid-model.mlmodel'
    with zipfile.ZipFile(model) as archive:
        manifest = json.loads(archive.read('manifest.json')); payload = archive.read('pipeline.skops')
    manifest['versions'] = {}
    with zipfile.ZipFile(bad, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest)); archive.writestr('pipeline.skops', payload)
    old_model = c.result; count = len(notices)
    QFileDialog.getOpenFileName = lambda *args: (str(bad), '')
    js("cmd('load_model');true"); wait_for(window, '!state.busy', timeout=60)
    assert any(n['kind'] == 'error' for n in notices[count:]) and c.result is old_model
    assert source.read_bytes() == original
    return ['Nominal cleaning preview/apply/undo/redo preserves source and agrees with issue counts',
            'Validated model saves/reopens and predicts without an active dataset',
            'Wrong-version evidence and incomplete model bundles report errors without stale output']
