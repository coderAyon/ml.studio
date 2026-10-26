"""Version checked model bundles; never execute a pickle or trust types from a file."""
import hashlib
import json
from pathlib import Path
import zipfile
from importlib.metadata import version
from .training import TrainingResult

# Explicitly audited application estimators, not names supplied by an archive.
TRUSTED={'mlstudio.training_prep.FoldIQRCapper','mlstudio.training_prep.BoundedFeatureSelector',
         'sklearn.feature_selection._univariate_selection.f_classif','sklearn.feature_selection._univariate_selection.f_regression',
         'numpy.dtype','sklearn.tree._tree.Tree','sklearn.metrics._dist_metrics.EuclideanDistance64',
         'sklearn.neighbors._kd_tree.KDTree'}


def save_model(path,result):
    import skops.io as sio
    data=sio.dumps(result.pipeline)
    manifest={'schema':1,'versions':{p:version(p) for p in ['numpy','scipy','scikit-learn','skops']},
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
            if set(z.namelist())!={'manifest.json','pipeline.skops'}:raise ValueError('Unsupported model bundle.')
            if sum(i.file_size for i in z.infolist())>500_000_000:raise ValueError('Expanded model exceeds the memory limit.')
            m=json.loads(z.read('manifest.json'));data=z.read('pipeline.skops')
        if m['schema']!=1 or hashlib.sha256(data).hexdigest()!=m['sha256']:raise ValueError('Model integrity check failed.')
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            if sum(i.file_size for i in inner.infolist())>500_000_000:raise ValueError('Expanded pipeline exceeds the memory limit.')
        if any(version(p)!=v for p,v in m['versions'].items()):raise ValueError('Model package versions differ. Use the original ML Studio environment or retrain.')
        unknown=set(sio.get_untrusted_types(data=data))
        if unknown-TRUSTED:raise ValueError('Unsupported model types: '+', '.join(sorted(unknown-TRUSTED)))
        pipeline=sio.loads(data,trusted=list(TRUSTED))
        if not hasattr(pipeline,'predict') or 'preprocess' not in pipeline.named_steps or 'model' not in pipeline.named_steps:raise ValueError('Invalid prediction pipeline.')
        return TrainingResult(pipeline,m['metrics'],m['metadata'])
    except (OSError,zipfile.BadZipFile,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('Invalid or unreadable ML Studio model bundle.') from exc
