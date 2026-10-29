"""One identity contract for read-only evidence from a recorded model holdout."""
import hashlib

import pandas as pd

from .data import schema_fingerprint


def validate_holdout(result, frame):
    """Validate before selecting rows or running inference; never fit any estimator."""
    from .training import TrainingResult
    if not isinstance(result, TrainingResult):
        raise ValueError('Train or open a saved model before inspecting diagnostics.')
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError('Open the original training dataset and version first.')
    meta = result.metadata
    if not isinstance(meta, dict) or meta.get('dataset_columns') != list(frame.columns):
        raise ValueError('Dataset column names or order do not match this model’s recorded schema. Restore the training schema, or train a new model to record it.')
    if meta.get('dataset_schema') != schema_fingerprint(frame):
        raise ValueError('Dataset types or category domains do not match the recorded schema. Restore the training types, or train a new model to record them.')
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if fingerprint != meta.get('dataset_fingerprint'):
        raise ValueError('This dataset differs from the training version. Open the same dataset and version used for this model before explaining inputs.')
    positions = meta.get('test_positions')
    if (not isinstance(positions, list) or not positions or len(positions) > 8000
            or any(type(p) is not int or not 0 <= p < len(frame) for p in positions)
            or len(positions) != len(set(positions)) or type(meta.get('test_rows')) is not int
            or len(positions) != meta['test_rows']):
        raise ValueError('This model has no valid recorded holdout manifest. Train a new model to create diagnostics.')
    features, target = meta.get('features'), meta.get('target')
    if (not isinstance(features, list) or not features or any(not isinstance(c, str) for c in features)
            or len(features) != len(set(features)) or not isinstance(target, str)
            or target not in frame or any(c not in frame for c in features) or target in features):
        raise ValueError('The saved model input and outcome columns do not match this dataset.')
    preprocess = result.pipeline.named_steps.get('preprocess')
    if list(getattr(preprocess, 'feature_names_in_', [])) != features:
        raise ValueError('The recorded feature schema differs from the fitted prediction pipeline.')
    if frame.iloc[positions][target].isna().any():
        raise ValueError('Recorded holdout outcomes contain missing values. Review the dataset and train again.')
    return features, target, positions
