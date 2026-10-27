"""A lightweight, independently animated launch window (no ML imports)."""
import math
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QWidget


class LaunchSplash(QWidget):
    def __init__(self, reduced_motion=False):
        super().__init__()
        self.setWindowTitle('ML Studio · Starting')
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(560, 350)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - self.rect().center())
        self.started = time.monotonic()
        self.stage = 'Starting your offline engine'
        self.phase = 0
        self.frames = 0
        self.reduced_motion = reduced_motion
        self.closing = None
        self.messages = queue.Queue()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33 if reduced_motion else 16)

    def tick(self):
        while not self.messages.empty():
            message = self.messages.get_nowait()
            if message == 'ready':
                self.stage = 'Your workspace is ready'
                self.phase = 2
                self.closing = time.monotonic()
            elif message == 'close':
                QApplication.quit()
                return
            elif message == 'activate':
                self.raise_()
                self.activateWindow()
            elif message.startswith('stage:'):
                self.stage = message[6:]
                self.phase = 1
        elapsed = time.monotonic() - self.started
        opacity = min(1., elapsed / .22)
        if self.closing is not None:
            opacity = max(0., 1. - (time.monotonic() - self.closing) / .25)
            if not opacity:
                QApplication.quit()
                return
        self.setWindowOpacity(1. if self.reduced_motion else opacity)
        self.frames += 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = 0. if self.reduced_motion else time.monotonic() - self.started
        # Soft outer shadow and a single, carefully bounded surface.
        for inset in range(8):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(14, 12, 40, 3 + inset))
            p.drawRoundedRect(QRectF(inset, inset + 3, 560 - 2*inset, 344 - 2*inset), 26, 26)
        surface = QRectF(8, 8, 544, 330)
        bg = QLinearGradient(8, 8, 552, 338)
        bg.setColorAt(0, QColor('#24233e'))
        bg.setColorAt(1, QColor('#121827'))
        p.setBrush(bg)
        p.setPen(QPen(QColor('#45415e'), 1))
        p.drawRoundedRect(surface, 22, 22)
        glow = QRadialGradient(QPointF(280, 100), 150)
        glow.setColorAt(0, QColor(130, 96, 255, 48))
        glow.setColorAt(1, QColor(130, 96, 255, 0))
        p.setBrush(glow)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(280, 100), 145, 90)

        # Orbiting signals hint at the app's visual workflow.
        center = QPointF(280, 97)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor('#423d60'), 1))
        p.drawEllipse(center, 72, 50)
        for i in range(3):
            angle = t * .7 + i * math.tau / 3
            point = QPointF(280 + 72*math.cos(angle), 97 + 50*math.sin(angle))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor('#b9a5ff'))
            p.drawEllipse(point, 3.2, 3.2)
        logo = QLinearGradient(250, 65, 310, 125)
        logo.setColorAt(0, QColor('#a78bff'))
        logo.setColorAt(1, QColor('#6940e2'))
        p.setBrush(logo)
        p.drawRoundedRect(QRectF(250, 67, 60, 60), 17, 17)
        font = QFont('Georgia', 29, QFont.Weight.Bold)
        font.setItalic(True)
        p.setFont(font)
        p.setPen(QColor('#ffffff'))
        p.drawText(QRectF(250, 64, 60, 64), Qt.AlignmentFlag.AlignCenter, 'ml')

        def label(rect, text, size, color, weight=QFont.Weight.Normal):
            p.setFont(QFont('Segoe UI', size, weight))
            p.setPen(QColor(color))
            p.drawText(QRectF(*rect), Qt.AlignmentFlag.AlignCenter, text)

        label((30, 157, 500, 40), 'ML Studio', 25, '#f5f2ff', QFont.Weight.DemiBold)
        label((30, 202, 500, 25), 'A little curiosity. Endless possibilities.', 10, '#b9bdd3')
        label((30, 246, 500, 24), self.stage + ('' if self.closing else '…'), 10, '#dfd9f5')

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor('#37354f'))
        p.drawRoundedRect(QRectF(136, 283, 288, 3), 1.5, 1.5)
        if self.closing:
            p.setBrush(QColor('#9c83ff'))
            p.drawRoundedRect(QRectF(136, 283, 288, 3), 1.5, 1.5)
        else:
            x = 136 + ((math.sin(t * 1.6) + 1) / 2) * 216
            gradient = QLinearGradient(x, 0, x + 72, 0)
            gradient.setColorAt(0, QColor('#8660ee'))
            gradient.setColorAt(.5, QColor('#d5c6ff'))
            gradient.setColorAt(1, QColor('#8660ee'))
            p.setBrush(gradient)
            p.drawRoundedRect(QRectF(x, 283, 72, 3), 1.5, 1.5)
        label((30, 300, 500, 21), 'LOCAL WORKSPACE   ·   ENTIRELY OFFLINE', 8, '#9ba3bd')
        p.end()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    reduced = False
    if sys.platform == 'win32':
        import ctypes
        enabled = ctypes.c_int(1)
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            reduced = not enabled.value
    splash = LaunchSplash(reduced)
    evidence = os.environ.get('MLSTUDIO_SPLASH_EVIDENCE')
    if evidence:
        destination = Path(evidence)
        destination.mkdir(parents=True, exist_ok=True)
        QTimer.singleShot(450, lambda: splash.grab().save(str(destination / 'launch-first.png')))
        def capture():
            splash.grab().save(str(destination / 'launch-animated.png'))
            (destination / 'animation.json').write_text(json.dumps({
                'pid': os.getpid(), 'frames': splash.frames,
                'reduced_motion': reduced, 'elapsed_seconds': round(time.monotonic()-splash.started, 2)
            }))
        QTimer.singleShot(1200, capture)

    def read_commands():
        try:
            for line in sys.stdin:
                splash.messages.put(line.strip())
        finally:
            # The pipe also closes when the owning IDE crashes or exits.
            splash.messages.put('close')

    threading.Thread(target=read_commands, daemon=True).start()
    splash.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
