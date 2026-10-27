"""Show launch feedback before ML imports and forward repeat launches to one window."""
import hashlib
import os
import sys
import time
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtNetwork import QLocalServer,QLocalSocket
from PySide6.QtWidgets import QApplication


def splash_message(app, message):
    process = getattr(app, 'mlstudio_splash', None)
    if process and process.poll() is None:
        try:
            process.stdin.write(message + '\n')
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass


def close_splash_if_running(app):
    process = getattr(app, 'mlstudio_splash', None)
    if process is None or process.poll() is not None:
        return
    splash_message(app, 'close')
    # The child owns the fade. This is only a recovery path if it misses IPC.
    QTimer.singleShot(350, lambda: process.terminate() if process.poll() is None else None)


def bootstrap(root):
    if QApplication.instance() is None:QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app=QApplication.instance() or QApplication(sys.argv)
    name='MLStudio-'+hashlib.sha256(str(Path(root).resolve()).lower().encode()).hexdigest()[:20]+os.environ.get('MLSTUDIO_TEST_INSTANCE','')
    socket=QLocalSocket();socket.connectToServer(name)
    if socket.waitForConnected(250):
        socket.write(b'activate');socket.waitForBytesWritten(500);socket.disconnectFromServer();return None
    server=QLocalServer(app);server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(name):
        socket.connectToServer(name)
        if socket.waitForConnected(1000):socket.write(b'activate');socket.waitForBytesWritten(500)
        return None
    # A separate lightweight Qt event loop stays responsive while ML imports
    # occupy the IDE's main thread. The private pipe owns the splash lifecycle.
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    splash = subprocess.Popen([sys.executable, str(Path(__file__).with_name('launch_splash.py'))],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, text=True, **options)
    app.mlstudio_splash = splash
    app.aboutToQuit.connect(lambda: splash_message(app, 'close'))
    def activate():
        while server.hasPendingConnections():
            client=server.nextPendingConnection();client.disconnectFromServer();client.deleteLater()
        windows=[w for w in app.topLevelWidgets() if w.objectName()=='MLStudioMain']
        if windows:
            window=windows[0]
            if window.isMinimized():window.showNormal()
            window.show();window.raise_();window.activateWindow()
        else:splash_message(app, 'activate')
        directory=os.environ.get('MLSTUDIO_STARTUP_EVIDENCE')
        if directory:
            import json
            path=Path(directory);path.mkdir(parents=True,exist_ok=True)
            (path/'activation.json').write_text(json.dumps({'pid':os.getpid(),'activated':True}))
    server.newConnection.connect(activate)
    app.mlstudio_server=server;app.mlstudio_splash=splash;app.mlstudio_started=time.perf_counter()
    return app


def ready(window):
    app=QApplication.instance();window.setObjectName('MLStudioMain')
    splash = getattr(app, 'mlstudio_splash', None)
    if splash is not None and splash.poll() is None:
        # Do not reveal the IDE behind its launch cover.
        window.hide()
    splash_message(app, 'stage:Preparing your workspace')
    # Wait for the actual first UI render, including its local WebChannel data.
    timer = QTimer(window)
    timer.setInterval(100)
    app.mlstudio_ready_timer = timer
    def check():
        window.view.page().runJavaScript("document.body?.dataset.ready === 'true'", loaded)
    def loaded(ok):
        if not timer.isActive():return
        if ok:
            timer.stop()
            splash_message(app, 'ready')
            # A stuck helper must never remain above the usable IDE window.
            delay = max(0, int((5.6 - (time.perf_counter()-app.mlstudio_started)) * 1000))
            QTimer.singleShot(delay, lambda: close_splash_if_running(app))
            reveal = QTimer(window)
            reveal.setInterval(40)
            def show_after_splash_exits():
                process = getattr(app, 'mlstudio_splash', None)
                if process is not None and process.poll() is None:
                    return
                reveal.stop()
                window.show()
                window.raise_()
                window.activateWindow()
                record_ready(app)
            reveal.timeout.connect(show_after_splash_exits)
            app.mlstudio_reveal_timer = reveal
            reveal.start()
            show_after_splash_exits()
        elif time.perf_counter()-app.mlstudio_started > 60:
            timer.stop()
            close_splash_if_running(app)
            window.show()
            window.raise_()
    timer.timeout.connect(check)
    timer.start()


def record_ready(app):
    directory=os.environ.get('MLSTUDIO_STARTUP_EVIDENCE')
    if directory:
        import json
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        splash = getattr(app, 'mlstudio_splash', None)
        (path/'ready.json').write_text(json.dumps({
            'pid': os.getpid(), 'splash_pid': splash.pid if splash else None,
            'splash_closed_at_ready': splash is None or splash.poll() is not None,
            'seconds_after_splash': time.perf_counter()-app.mlstudio_started
        }))
