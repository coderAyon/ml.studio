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
            record_ready(app)
        elif time.perf_counter()-app.mlstudio_started > 60:
            timer.stop()
            splash_message(app, 'close')
    timer.timeout.connect(check)
    timer.start()


def record_ready(app):
    directory=os.environ.get('MLSTUDIO_STARTUP_EVIDENCE')
    if directory:
        import json
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        (path/'ready.json').write_text(json.dumps({'pid':os.getpid(),'seconds_after_splash':time.perf_counter()-app.mlstudio_started}))
