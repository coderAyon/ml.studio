"""Stable project-local EXE. Loads current source without rebuilding the app."""
from pathlib import Path
import ctypes
import os
import subprocess
import sys


def main():
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    python = root / ".venv/Scripts/pythonw.exe"
    script = root / "main.py"
    if not python.is_file() or not script.is_file():
        ctypes.windll.user32.MessageBoxW(None, "Keep MLStudio.exe in the project folder with main.py, src and .venv. This launcher runs the current local source.", "ML Studio — workspace missing", 16)
        return 1
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    # PyInstaller modifies DLL resolution; reset it for the source interpreter.
    if getattr(sys, "frozen", False):
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    subprocess.Popen([str(python), str(script), *sys.argv[1:]], cwd=root, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
