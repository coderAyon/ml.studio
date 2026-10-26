# Rebuilding this launcher is unnecessary when editing app source or UI assets.
from pathlib import Path
root = Path(SPECPATH).parent
a = Analysis([str(root/'packaging/workspace_launcher.py')], pathex=[], binaries=[], datas=[], hiddenimports=[], hookspath=[], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,a.binaries,a.datas,[],name='MLStudio',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False)
