"""Lossless persistence must work through projects and independently replayed studies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from mlstudio.data import DatasetVersions, Transformation, load_csv
from mlstudio.research import ResearchConfig, ResearchSession
from mlstudio.research_export import export_pack


def replay_in_separate_process(directory):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    script = """from pathlib import Path
from mlstudio.research_export import replay_pack
replay_pack(Path.cwd())
import sys
root=Path.cwd().resolve()
assert all(Path(m.__file__).resolve().is_relative_to(root) for n,m in sys.modules.items()
           if n=='mlstudio' or n.startswith('mlstudio.'))
"""
    return subprocess.run([sys.executable, '-c', script], cwd=directory, env=environment,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('frequency,timezone,ties', [('us', None, 1), ('ns', None, 2), ('ns', 'Asia/Dhaka', 1)])
def test_time_research_pack_preserves_submillisecond_splits(tmp_path, frequency, timezone, ties):
    times = pd.date_range('2026-01-01', periods=100 // ties, freq=frequency, tz=timezone).repeat(ties)
    frame = pd.DataFrame({'time': times, 'x': np.arange(100, dtype=float), 'label': [0, 1] * 50})
    session = ResearchSession(frame, ResearchConfig(target='label', protocol='time', time_column='time', folds=2, repeats=1))
    session.compare(['Decision Tree'])
    archive = tmp_path / 'study.zip'
    export_pack(archive, session, {'fixture': 'synthetic'})
    directory = tmp_path / 'replay'
    with zipfile.ZipFile(archive) as pack:
        manifest = json.loads(pack.read('manifest.json'))
        assert manifest['schema'] == 2
        assert 'mlstudio/project_data.py' in pack.namelist()
        from mlstudio.project_data import unpack_frame
        restored = unpack_frame(json.loads(pack.read('frame.json')))
        pd.testing.assert_frame_equal(session.frame, restored)
        assert restored.time.nunique() == 100 // ties
        pack.extractall(directory)  # Our own generated pack only.
    process = replay_in_separate_process(directory)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((directory / 'reproduced.json').read_text())['verified']


def test_legacy_research_table_json_pack_remains_readable(tmp_path):
    frame = pd.DataFrame({'x': np.arange(80, dtype=float), 'label': [0, 1] * 40})
    session = ResearchSession(frame, ResearchConfig(target='label', folds=2, repeats=1))
    session.compare(['Decision Tree'])
    archive = tmp_path / 'study.zip'
    export_pack(archive, session, {})
    directory = tmp_path / 'legacy'
    with zipfile.ZipFile(archive) as pack:
        pack.extractall(directory)
    typed = session.frame.to_json(orient='table', index=False, date_format='iso', double_precision=15).encode()
    (directory / 'frame.json').write_bytes(typed)
    manifest = json.loads((directory / 'manifest.json').read_text())
    manifest.update(schema=1, typed_data_sha256=hashlib.sha256(typed).hexdigest())
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    process = replay_in_separate_process(directory)
    assert process.returncode == 0, process.stdout + process.stderr
    manifest['schema'] = 999
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    process = replay_in_separate_process(directory)
    assert process.returncode != 0 and 'schema' in process.stderr.lower()


def test_failed_research_snapshot_preserves_existing_destination(tmp_path):
    from types import SimpleNamespace
    destination = tmp_path / 'existing.zip'
    destination.write_bytes(b'previous user export')
    session = SimpleNamespace(frame=pd.DataFrame({'unsupported': [object()]}))
    with pytest.raises(ValueError, match='Unsupported|unsupported'):
        export_pack(destination, session, {})
    assert destination.read_bytes() == b'previous user export'


def test_project_save_reopen_without_source_preserves_unsigned_values(tmp_path, monkeypatch):
    from test_bridge import controller_with_data
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QFileDialog
    app, controller, bridge = controller_with_data(tmp_path)
    source = tmp_path / 'unsigned.csv'
    source.write_text('identifier,label\n0, A\n18446744073709551615,b\n', encoding='utf-8')
    original = source.read_bytes()
    controller.source = str(source)
    controller.import_options = {}
    controller.dataset = DatasetVersions(load_csv(source))
    controller.dataset.apply(Transformation('Trim whitespace', ['label']))
    expected = controller.dataset.current
    controller.refresh_data()
    project = tmp_path / 'portable.json'
    controller.save_project(str(project))
    assert source.read_bytes() == original
    source.unlink()  # Only a synthetic temporary source, to exercise portability.
    errors = []
    controller.error_handler = errors.append
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(project), ''))
    controller.open_project()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if not controller.busy else None)
    timer.start(10)
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    timer.stop()
    assert not controller.busy and not errors
    pd.testing.assert_frame_equal(controller.dataset.current, expected)
    controller.dataset.undo()
    assert controller.dataset.current.label.iloc[0] == ' A'
    controller.dataset.redo()
    output = tmp_path / 'restored.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
    controller.export_csv()
    assert '18446744073709551615' in output.read_text()
    controller.close()
