import collections

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class LineGraphWidget(QWidget):
    """Dynamic line graph widget matching the Cyber reference UI."""

    def __init__(self, history_size: int = 100) -> None:
        super().__init__()
        self._history = collections.deque([0.0] * history_size, maxlen=history_size)
        self.setMinimumHeight(80)

    def set_value(self, value: float) -> None:
        import random

        jitter = random.uniform(-0.5, 0.5) if value > 90 else 0.0
        self._history.append(min(100.0, max(0.0, value + jitter)))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.setBrush(QColor("#1E232D"))
        painter.setPen(QPen(QColor("#3A475C"), 1))
        painter.drawRoundedRect(0, 0, w, h, 6, 6)
        if len(self._history) < 2:
            return
        path = QPainterPath()
        dx = w / (len(self._history) - 1)
        pad_y = 10
        usable_h = h - (pad_y * 2)
        first_val = self._history[0] / 100.0
        path.moveTo(0, h - pad_y - (first_val * usable_h))
        for i, val in enumerate(self._history):
            if i == 0:
                continue
            normalized_val = val / 100.0
            x = i * dx
            y = h - pad_y - (normalized_val * usable_h)
            path.lineTo(x, y)
        pen = QPen(QColor("#00FFcc"), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        fill_path = QPainterPath(path)
        fill_path.lineTo(w, h - pad_y)
        fill_path.lineTo(0, h - pad_y)
        fill_path.closeSubpath()
        fill_grad = QLinearGradient(0, pad_y, 0, h - pad_y)
        fill_grad.setColorAt(0, QColor(57, 255, 20, 180))
        fill_grad.setColorAt(1, QColor(255, 42, 42, 180))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fill_grad))
        painter.drawPath(fill_path)
