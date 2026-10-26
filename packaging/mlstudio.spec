# Full portable distribution; project development uses MLStudio.exe launcher instead.
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata,collect_submodules
root = Path(SPECPATH).parent
replay_files = ['__init__.py', 'training.py', 'training_prep.py', 'model_settings.py', 'evaluation.py', 'research.py', 'safety.py', 'error_analysis.py', 'research_export.py']
resources = [(str(root / 'src/mlstudio/web'), 'mlstudio/web')]
resources += [(str(root / 'src/mlstudio' / name), 'replay_sources') for name in replay_files]
for package in ['numpy', 'pandas', 'scipy', 'scikit-learn','skops']:
    resources += copy_metadata(package)
a = Analysis([str(root / 'main.py')], pathex=[str(root / 'src')], binaries=[], datas=resources, hiddenimports=['matplotlib.backends.backend_svg',*collect_submodules('skops.io'),'skops.io.old._general_v0','skops.io.old._numpy_v0','skops.io.old._numpy_v1'], hookspath=[], excludes=['PySide6.QtWebEngineQuick', 'tkinter', 'torch', 'pytest'], noarchive=False)
a.binaries = [entry for entry in a.binaries if entry[0].lower() not in {'icuuc.dll', 'icudt78.dll'}]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MLStudio', debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MLStudio-Research')
