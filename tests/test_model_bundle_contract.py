"""Only our own fitted skops fixtures are opened; no external pickle is used."""
import copy
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from mlstudio.model_io import load_model, save_model
from mlstudio.training import predict, train


@pytest.fixture
def model_bundle(tmp_path):
    frame = pd.DataFrame({'x': np.arange(80, dtype=float), 'z': np.sin(np.arange(80)),
                          'label': [0, 1] * 40})
    result = train(frame, 'label', 'Decision Tree')
    path = tmp_path / 'original.mlmodel'; save_model(path, result)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        pipeline = archive.read('pipeline.skops')
    return frame, result, path, manifest, pipeline


@pytest.mark.parametrize('change', ['empty_versions', 'missing_version', 'extra_version', 'empty_metadata',
    'metadata_list', 'empty_metrics', 'metrics_list', 'boolean_schema', 'features_reordered',
    'features_missing', 'features_duplicate', 'target_is_feature', 'unknown_algorithm',
    'wrong_estimator', 'wrong_parameter', 'invalid_positions', 'wrong_test_count',
    'bad_fingerprint', 'missing_schema', 'wrong_seed', 'nan_metric', 'duplicate_entry'])
def test_model_bundle_rejects_incomplete_or_inconsistent_manifests(model_bundle, tmp_path, change):
    _, _, _, m, pipeline = model_bundle
    if change == 'empty_versions': m['versions'] = {}
    elif change == 'missing_version': del m['versions']['numpy']
    elif change == 'extra_version': m['versions']['not-a-package'] = '1'
    elif change == 'empty_metadata': m['metadata'] = {}
    elif change == 'metadata_list': m['metadata'] = []
    elif change == 'empty_metrics': m['metrics'] = {}
    elif change == 'metrics_list': m['metrics'] = []
    elif change == 'boolean_schema': m['schema'] = True
    elif change == 'features_reordered': m['metadata']['features'] = ['z', 'x']
    elif change == 'features_missing': m['metadata']['features'] = ['x']
    elif change == 'features_duplicate': m['metadata']['features'] = ['x', 'x']
    elif change == 'target_is_feature': m['metadata']['target'] = 'x'
    elif change == 'unknown_algorithm': m['metadata']['algorithm'] = 'fabricated'
    elif change == 'wrong_estimator': m['metadata']['algorithm'] = 'Logistic Regression'
    elif change == 'wrong_parameter': m['metadata']['parameters']['max_depth'] = 999
    elif change == 'invalid_positions': m['metadata']['test_positions'][0] = True
    elif change == 'wrong_test_count': m['metadata']['test_rows'] += 1
    elif change == 'bad_fingerprint': m['metadata']['dataset_fingerprint'] = ''
    elif change == 'missing_schema': del m['metadata']['dataset_schema']
    elif change == 'wrong_seed': m['metadata']['seed'] = 100
    elif change == 'nan_metric': m['metrics']['Accuracy'] = float('nan')
    path = tmp_path / 'altered.mlmodel'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(m))
        archive.writestr('pipeline.skops', pipeline)
        if change == 'duplicate_entry':
            with pytest.warns(UserWarning, match='Duplicate name'):
                archive.writestr('manifest.json', json.dumps(m))
    with pytest.raises(ValueError): load_model(path)


def test_model_roundtrip_preserves_evidence_and_rejects_bad_resave_atomically(model_bundle):
    frame, result, path, _, _ = model_bundle
    loaded = load_model(path)
    assert loaded.metadata == result.metadata
    np.testing.assert_array_equal(predict(loaded, frame[['x', 'z']]), predict(result, frame[['x', 'z']]))
    original = path.read_bytes()
    result.metadata = copy.deepcopy(result.metadata); result.metadata['features'] = ['z', 'x']
    with pytest.raises(ValueError): save_model(path, result)
    assert path.read_bytes() == original


def test_unfitted_estimator_cannot_be_saved_as_a_trained_model(model_bundle):
    from sklearn.base import clone
    _, result, path, _, _ = model_bundle
    original = path.read_bytes()
    result.pipeline.steps[-1] = ('model', clone(result.pipeline.named_steps['model']))
    with pytest.raises(ValueError): save_model(path, result)
    assert path.read_bytes() == original


def test_model_loader_rejects_unfitted_final_estimator(model_bundle, tmp_path):
    import hashlib
    import skops.io as sio
    from sklearn.base import clone
    _, result, _, manifest, _ = model_bundle
    result.pipeline.steps[-1] = ('model', clone(result.pipeline.named_steps['model']))
    payload = sio.dumps(result.pipeline); manifest['sha256'] = hashlib.sha256(payload).hexdigest()
    path = tmp_path / 'unfitted.mlmodel'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest)); archive.writestr('pipeline.skops', payload)
    with pytest.raises(ValueError): load_model(path)


def test_model_loader_rejects_overflowing_json_numbers_in_secondary_metrics(model_bundle, tmp_path):
    _, _, _, manifest, payload = model_bundle
    manifest['metrics']['secondary'] = 'overflow-placeholder'
    text = json.dumps(manifest).replace('"overflow-placeholder"', '1e999')
    path = tmp_path / 'overflow.mlmodel'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', text); archive.writestr('pipeline.skops', payload)
    with pytest.raises(ValueError): load_model(path)


@pytest.mark.parametrize('step', ['imputer', 'scaler'])
def test_partly_unfitted_numeric_preparation_cannot_be_saved_or_loaded(model_bundle, tmp_path, step):
    import hashlib
    import skops.io as sio
    from sklearn.base import clone
    _, result, path, manifest, _ = model_bundle
    original = path.read_bytes()
    numeric = result.pipeline.named_steps['preprocess'].named_transformers_['numeric']
    index = next(i for i, (name, _) in enumerate(numeric.steps) if name == step)
    numeric.steps[index] = (step, clone(numeric.named_steps[step]))
    with pytest.raises(ValueError): save_model(path, result)
    assert path.read_bytes() == original
    payload = sio.dumps(result.pipeline); manifest['sha256'] = hashlib.sha256(payload).hexdigest()
    path = tmp_path / 'partial.mlmodel'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest)); archive.writestr('pipeline.skops', payload)
    with pytest.raises(ValueError): load_model(path)
