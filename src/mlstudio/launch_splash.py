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
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget


class LaunchSplash(QWidget):
    def __init__(self, reduced_motion=False):
        super().__init__()
        self.setWindowTitle('ML Studio · Starting')
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(660, 350)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - self.rect().center())
        self.started = time.monotonic()
        self.stage = 'Opening your workspace'
        self.phase = 0
        self.frames = 0
        self.reduced_motion = reduced_motion
        self.closing = None
        self.ready_received = None
        self.minimum_visible_seconds = 5.0
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
                if self.ready_received is None:
                    self.ready_received = time.monotonic()
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
        if (self.ready_received is not None and self.closing is None
                and elapsed >= self.minimum_visible_seconds):
            self.closing = time.monotonic()
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

        def text(rect, value, size, color, weight=QFont.Weight.Normal,
                 align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter):
            font = QFont('Segoe UI')
            font.setPixelSize(size)
            font.setWeight(weight)
            p.setFont(font)
            p.setPen(QColor(color))
            p.drawText(QRectF(*rect), align, value)

        # A landscape launch card with a quiet, nearly square silhouette.
        p.setPen(Qt.PenStyle.NoPen)
        for inset in range(8):
            p.setBrush(QColor(35, 26, 65, 2 + inset))
            p.drawRoundedRect(QRectF(inset, inset + 3, 660-2*inset, 344-2*inset), 13, 13)
        background = QLinearGradient(8, 8, 652, 342)
        background.setColorAt(0, QColor('#ffffff'))
        background.setColorAt(1, QColor('#fbfaff'))
        p.setBrush(background)
        p.setPen(QPen(QColor('#dedde9'), 1))
        p.drawRoundedRect(QRectF(8, 8, 644, 334), 10, 10)

        # Wordmark and left-aligned type replace the old central orbit.
        logo = QLinearGradient(36, 70, 90, 126)
        logo.setColorAt(0, QColor('#9c7cff'))
        logo.setColorAt(1, QColor('#6741d2'))
        p.setBrush(logo)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(36, 70, 54, 56), 12, 12)
        font = QFont('Georgia')
        font.setPixelSize(37)
        font.setBold(True)
        font.setItalic(True)
        p.setFont(font)
        p.setPen(QColor('#ffffff'))
        p.drawText(QRectF(36, 70, 54, 56), Qt.AlignmentFlag.AlignCenter, 'ml')
        text((105, 70, 205, 56), 'ML Studio', 34, '#202338', QFont.Weight.DemiBold)
        text((36, 151, 275, 24), 'Your offline machine learning workspace.', 13, '#687086')

        # A decorative miniature workflow, without pretend task completion.
        p.setPen(Qt.PenStyle.NoPen)
        panel = QLinearGradient(326, 26, 626, 318)
        panel.setColorAt(0, QColor('#f3efff'))
        panel.setColorAt(1, QColor('#eeeffa'))
        p.setBrush(panel)
        p.drawRoundedRect(QRectF(326, 26, 300, 292), 8, 8)
        positions = [(348, 113), (444, 159), (540, 113)]
        arrival_offsets = [(-10, -52), (44, 64), (20, -60)]
        arrival_duration = .88
        arrival_gap = .46

        def arrival(index):
            if self.reduced_motion:
                return 1.
            progress = (t - (.22 + index * arrival_gap)) / arrival_duration
            progress = min(1., max(0., progress))
            # Cubic ease-out gives each tile a short, soft landing.
            return 1. - (1. - progress) ** 3

        node_progress = [arrival(index) for index in range(3)]
        rendered_positions = [
            (x + arrival_offsets[index][0] * (1. - node_progress[index]),
             y + arrival_offsets[index][1] * (1. - node_progress[index]))
            for index, (x, y) in enumerate(positions)
        ]
        for index in range(2):
            x, y = rendered_positions[index]
            nx, ny = rendered_positions[index+1]
            path = QPainterPath(QPointF(x+66, y+34))
            path.cubicTo(QPointF(x+86, y+34), QPointF(nx-20, ny+34), QPointF(nx, ny+34))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor('#cec5e8'), 1.5))
            reveal = min(node_progress[index], node_progress[index + 1])
            if reveal > 0:
                segment = QPainterPath(path.pointAtPercent(0))
                steps = max(2, int(36 * reveal))
                for step in range(1, steps + 1):
                    segment.lineTo(path.pointAtPercent(reveal * step / steps))
                p.drawPath(segment)
            if not self.reduced_motion:
                lead = (t - (.22 + index * arrival_gap)) / arrival_duration
                lead = min(reveal, max(0., lead))
                pen = QPen(QColor('#9670ed'), 3)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                if lead > 0:
                    p.drawLine(path.pointAtPercent(max(0., lead-.10)), path.pointAtPercent(lead))

        for index, ((x, y), name) in enumerate(zip(positions, ['Data', 'Prepare', 'Model'])):
            progress = node_progress[index]
            x, y = rendered_positions[index]
            p.save()
            p.setOpacity(.28 + .72 * progress)
            pulse = 0. if self.reduced_motion or progress < 1 else (1+math.sin((t-2.1)*2.3-index*1.1))/2
            p.setBrush(QColor('#ffffff'))
            border = QColor('#d9d0ed')
            border.setAlphaF(.55+.45*pulse)
            p.setPen(QPen(border, 1))
            p.drawRoundedRect(QRectF(x, y, 66, 68), 9, 9)
            p.setPen(QPen(QColor('#8054d5'), 1.3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            ix, iy = x+24, y+13
            if index == 0:
                p.drawRoundedRect(QRectF(ix, iy, 19, 17), 2, 2)
                p.drawLine(QPointF(ix, iy+6), QPointF(ix+19, iy+6))
                p.drawLine(QPointF(ix+7, iy), QPointF(ix+7, iy+17))
                p.drawLine(QPointF(ix, iy+12), QPointF(ix+19, iy+12))
            elif index == 1:
                for row, offset in enumerate([5, 14, 8]):
                    cy = iy+2+row*6
                    p.drawLine(QPointF(ix, cy), QPointF(ix+19, cy))
                    p.setBrush(QColor('#ffffff'))
                    p.drawRoundedRect(QRectF(ix+offset-2, cy-2, 4, 4), 1, 1)
            else:
                for a, b in [((0, 9), (9, 1)), ((0, 9), (9, 17)), ((9, 1), (19, 9)), ((9, 17), (19, 9))]:
                    p.drawLine(QPointF(ix+a[0], iy+a[1]), QPointF(ix+b[0], iy+b[1]))
                p.setBrush(QColor('#ffffff'))
                for dx, dy in [(0, 9), (9, 1), (9, 17), (19, 9)]:
                    p.drawRect(QRectF(ix+dx-2, iy+dy-2, 4, 4))
            text((x+3, y+40, 60, 19), name, 10, '#55466e', QFont.Weight.DemiBold,
                 Qt.AlignmentFlag.AlignCenter)
            p.restore()

        # Small abstract chart strokes give the composition a data texture.
        p.setPen(QPen(QColor('#d8d2e8'), 1))
        p.drawLine(QPointF(350, 282), QPointF(602, 282))
        p.setPen(Qt.PenStyle.NoPen)
        for index, height in enumerate([9, 15, 23, 30, 22, 14, 19, 27, 17, 11]):
            alpha = .35 if self.reduced_motion else .3+.23*(1+math.sin(t*1.7-index*.4))/2
            color = QColor('#8d70cf')
            color.setAlphaF(alpha)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(355+index*25, 281-height, 15, height), 2, 2)

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
