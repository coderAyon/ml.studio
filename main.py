from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
if __name__ == "__main__":
    try:
        if not any(arg.endswith('smoke') or arg=='--contest-audit' for arg in sys.argv[1:]):
            from mlstudio.startup import bootstrap
            if bootstrap(Path(__file__).parent) is None:raise SystemExit(0)
        from mlstudio.app import main
        main()
    except Exception:
        import os,traceback
        try:
            from PySide6.QtWidgets import QApplication
            from mlstudio.startup import splash_message
            if QApplication.instance():splash_message(QApplication.instance(), 'close')
        except Exception:
            pass  # Preserve the original startup error even if Qt itself failed.
        directory=Path(os.environ.get('MLSTUDIO_HOME') or os.environ.get('LOCALAPPDATA') or Path.home())/'MLStudio'
        directory.mkdir(parents=True,exist_ok=True);log=directory/'startup-error.log';log.write_text(traceback.format_exc(),encoding='utf-8')
        if sys.platform=='win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,'ML Studio could not start. Details were saved to:\n'+str(log),'ML Studio · Startup error',16)
        raise
