"""Version checked model bundles; never execute a pickle or trust types from a file."""
import hashlib
import json
from pathlib import Path
import zipfile
import math
import re
from importlib.metadata import version
from .training import ALGORITHMS, REGRESSION, TrainingResult

PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'skops')


def _names(values):
    return (isinstance(values, list) and bool(values)
            and all(isinstance(v, str) and bool(v) for v in values)
            and len(values) == len(set(values)))


def _check_fitted_components(estimator):
    """Check active pipeline steps; empty ColumnTransformer branches are unfitted by design."""
    from sklearn.utils.validation import check_is_fitted
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from .training_prep import BoundedFeatureSelector
    check_is_fitted(estimator)
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            if step is not None and not isinstance(step, str):
                _check_fitted_components(step)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, columns in estimator.transformers_:
            if len(columns) and not isinstance(transformer, str):
                _check_fitted_components(transformer)
    elif isinstance(estimator, BoundedFeatureSelector):
        _check_fitted_components(estimator.selector_)


def _validate_record(metrics, meta, pipeline=None):
    """Validate evidence before save/load; never infer missing provenance."""
    if not isinstance(meta, dict) or not isinstance(metrics, dict) or not metrics:
        raise ValueError('Missing model metadata or metrics. Retrain and save a new model.')
    features, columns, target = meta.get('features'), meta.get('dataset_columns'), meta.get('target')
    if (not _names(features) or len(features) > 200 or not _names(columns)
            or not isinstance(target, str) or target not in columns or target in features
            or not set(features) <= set(columns)):
        raise ValueError('Invalid saved model input/outcome schema. Retrain and save a new model.')
    algorithm = meta.get('algorithm')
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
        raise ValueError('Unsupported recorded model algorithm.')
    if any(not isinstance(meta.get(key), str) or not re.fullmatch('[0-9a-f]{64}', meta[key])
           for key in ('dataset_fingerprint', 'dataset_schema')):
        raise ValueError('Missing or invalid dataset identity. Retrain and save a new model.')
    training, testing, share = meta.get('training_rows'), meta.get('test_rows'), meta.get('test_size')
    if (type(training) is not int or type(testing) is not int or min(training, testing) < 1
            or not 8 <= training + testing <= 20000 or type(share) not in (float, int)
            or not math.isfinite(share) or not .1 <= share <= .4
            or testing != math.ceil((training + testing) * share)):
        raise ValueError('Invalid recorded train/test split.')
    positions = meta.get('test_positions')
    if (not isinstance(positions, list) or len(positions) != testing
            or any(type(p) is not int or not 0 <= p < training + testing for p in positions)
            or len(set(positions)) != testing):
        raise ValueError('Invalid recorded holdout manifest.')
    seed = meta.get('seed')
    if type(seed) is not int or not 0 <= seed < 2**32 or not isinstance(meta.get('parameters'), dict):
        raise ValueError('Invalid recorded model parameters or seed.')
    from .training_prep import validate_preparation
    validate_preparation(meta.get('preparation'), columns, target)
    primary = metrics.get('RMSE' if algorithm in REGRESSION else 'Accuracy')
    if (type(primary) not in (int, float) or not math.isfinite(primary) or primary < 0
            or (algorithm not in REGRESSION and primary > 1)):
        raise ValueError('Missing or invalid task metrics.')
    if pipeline is None:
        return
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    if not isinstance(pipeline, Pipeline) or not {'preprocess', 'model'} <= set(pipeline.named_steps):
        raise ValueError('Invalid prediction pipeline.')
    preprocess, model = pipeline.named_steps['preprocess'], pipeline.named_steps['model']
    if (not isinstance(preprocess, ColumnTransformer)
            or list(getattr(preprocess, 'feature_names_in_', [])) != features
            or type(model) is not ALGORITHMS[algorithm]
            or not hasattr(pipeline, 'predict')):
        raise ValueError('Recorded input schema or algorithm differs from the fitted pipeline.')
    from sklearn.exceptions import NotFittedError
    try:
        _check_fitted_components(pipeline)
    except NotFittedError as exc:
        raise ValueError('The saved pipeline contains an unfitted estimator. Train a model before saving.') from exc
    # Compare canonical JSON because JSON converts estimator tuple parameters to lists.
    if json.dumps(model.get_params(), sort_keys=True, allow_nan=False) != json.dumps(meta['parameters'], sort_keys=True, allow_nan=False):
        raise ValueError('Recorded parameters differ from the fitted model.')
    if 'random_state' in model.get_params() and model.random_state != seed:
        raise ValueError('Recorded seed differs from the fitted model.')


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate model manifest field.')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite values are not allowed in model manifests.')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _invalid_constant(value)
    return result

# Explicitly audited application estimators, not names supplied by an archive.
TRUSTED={'mlstudio.training_prep.FoldIQRCapper','mlstudio.training_prep.BoundedFeatureSelector',
         'sklearn.feature_selection._univariate_selection.f_classif','sklearn.feature_selection._univariate_selection.f_regression',
         'numpy.dtype','sklearn.tree._tree.Tree','sklearn.metrics._dist_metrics.EuclideanDistance64',
         'sklearn.neighbors._kd_tree.KDTree'}


def save_model(path,result):
    import skops.io as sio
    _validate_record(result.metrics, result.metadata, result.pipeline)
    data=sio.dumps(result.pipeline)
    manifest={'schema':1,'versions':{p:version(p) for p in PACKAGES},
              'sha256':hashlib.sha256(data).hexdigest(),'metrics':result.metrics,'metadata':result.metadata}
    from uuid import uuid4
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json',json.dumps(manifest,allow_nan=False));z.writestr('pipeline.skops',data)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


def load_model(path):
    import skops.io as sio
    try:
        path=Path(path)
        if path.stat().st_size>250_000_000:raise ValueError('Model file exceeds 250 MB.')
        with zipfile.ZipFile(path) as z:
            if len(z.namelist()) != 2 or set(z.namelist())!={'manifest.json','pipeline.skops'}:raise ValueError('Unsupported model bundle.')
            if sum(i.file_size for i in z.infolist())>500_000_000:raise ValueError('Expanded model exceeds the memory limit.')
            if z.getinfo('manifest.json').file_size > 10_000_000:raise ValueError('Model manifest exceeds 10 MB.')
            m=json.loads(z.read('manifest.json'), object_pairs_hook=_json_pairs,
                         parse_constant=_invalid_constant, parse_float=_finite_float)
            if (not isinstance(m, dict) or set(m) != {'schema','versions','sha256','metrics','metadata'}
                    or type(m['schema']) is not int or m['schema'] != 1):
                raise ValueError('Unsupported model manifest. Retrain and save a new model.')
            if not isinstance(m['versions'], dict) or set(m['versions']) != set(PACKAGES):
                raise ValueError('Model package version manifest is incomplete or unsupported.')
            if any(not isinstance(m['versions'][p],str) or version(p)!=m['versions'][p] for p in PACKAGES):
                raise ValueError('Model package versions differ. Use the original ML Studio environment or retrain.')
            _validate_record(m['metrics'], m['metadata'])
            data=z.read('pipeline.skops')
        if not isinstance(m['sha256'], str) or hashlib.sha256(data).hexdigest()!=m['sha256']:raise ValueError('Model integrity check failed.')
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            if sum(i.file_size for i in inner.infolist())>500_000_000:raise ValueError('Expanded pipeline exceeds the memory limit.')
        unknown=set(sio.get_untrusted_types(data=data))
        if unknown-TRUSTED:raise ValueError('Unsupported model types: '+', '.join(sorted(unknown-TRUSTED)))
        pipeline=sio.loads(data,trusted=list(TRUSTED))
        _validate_record(m['metrics'], m['metadata'], pipeline)
        return TrainingResult(pipeline,m['metrics'],m['metadata'])
    except (OSError,zipfile.BadZipFile,KeyError,TypeError,UnicodeError,RecursionError,json.JSONDecodeError) as exc:
        raise ValueError('Invalid or unreadable ML Studio model bundle.') from exc
