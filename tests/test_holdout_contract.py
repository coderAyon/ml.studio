"""Every explanation consumer must reject evidence from a different schema/version."""
import copy
import numpy as np
import pandas as pd
import pytest

from mlstudio.evaluation import explain_inputs
from mlstudio.model_diagnostics import build_diagnostics
from mlstudio.training import train


@pytest.fixture
def experiment():
    frame = pd.DataFrame({'x': np.arange(80, dtype=float),
                          'z': np.sin(np.arange(80)), 'label': [0, 1] * 40})
    return frame, train(frame, 'label', 'Decision Tree')


@pytest.mark.parametrize('consumer', [explain_inputs, build_diagnostics])
@pytest.mark.parametrize('change', ['names', 'order', 'dtype', 'values', 'positions',
                                    'duplicate_positions', 'holdout_count', 'features'])
def test_explanation_rejects_mismatched_evidence(experiment, consumer, change):
    frame, result = experiment
    if change == 'names':
        frame = frame.rename(columns={'x': 'z', 'z': 'x'})
    elif change == 'order':
        frame = frame[['z', 'x', 'label']]
    elif change == 'dtype':
        frame = frame.astype({'x': 'int64'})
    elif change == 'values':
        frame = frame.copy(); frame.loc[0, 'x'] = 999
    elif change == 'positions':
        result.metadata['test_positions'] = [True, *result.metadata['test_positions'][1:]]
    elif change == 'duplicate_positions':
        result.metadata['test_positions'][0] = result.metadata['test_positions'][1]
    elif change == 'holdout_count':
        result.metadata['test_rows'] += 1
    else:
        result.metadata['features'] = ['z', 'x']
    with pytest.raises(ValueError):
        consumer(result, frame)


def test_nominal_domain_is_part_of_explanation_identity():
    frame = pd.DataFrame({'x': range(80), 'city': pd.Categorical(['a', 'b'] * 40),
                          'label': [0, 1] * 40})
    result = train(frame, 'label', 'Decision Tree')
    changed = frame.copy(); changed['city'] = changed.city.cat.add_categories(['unused'])
    with pytest.raises(ValueError):
        explain_inputs(result, changed)


def test_input_explanation_is_read_only_and_uses_recorded_rows(experiment, monkeypatch):
    frame, result = experiment
    original = frame.copy(deep=True); metadata = copy.deepcopy(result.metadata)
    monkeypatch.setattr(result.pipeline, 'fit', lambda *a, **k: pytest.fail('Must not refit'))
    report = explain_inputs(result, frame)
    assert report['sample_rows'] == 20
    assert {r['feature'] for r in report['rows']} == {'x', 'z'}
    assert all(np.isfinite(r['mean']) and np.isfinite(r['std']) for r in report['rows'])
    assert result.metadata == metadata
    pd.testing.assert_frame_equal(frame, original)


def test_prediction_workspace_clears_explanation_after_dataset_revision_change(tmp_path, experiment):
    from test_bridge import controller_with_data
    from PySide6.QtCore import QEventLoop, QTimer
    from mlstudio.data import DatasetVersions, Transformation
    app, controller, bridge = controller_with_data(tmp_path)
    frame, result = experiment
    controller.dataset = DatasetVersions(frame); controller.result = result
    errors = []; controller.error_handler = errors.append
    bridge.predictions.command('prediction_explain', {})
    while controller.busy:
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    assert not errors and bridge.predictions.state()['explanation']['sample_rows'] == 20
    controller.dataset.apply(Transformation('Rename column', ['x'], 'renamed'))
    assert bridge.predictions.state()['explanation'] is None
    assert controller.result is result  # Prediction can still use the fitted model.
    controller.close()
