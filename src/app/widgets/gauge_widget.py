from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CircularGauge(QWidget):
    """A circular gauge for displaying normalized metrics (EAR, MAR, PERCLOS)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(120, 140)
        self._title = title
        self._value = 0.0
        self._threshold = 0.5
        self._inverse = False

    def set_value(self, value: float) -> None:
        self._value = min(1.0, max(0.0, value))
        self.update()

    def set_threshold(self, threshold: float, inverse: bool = False) -> None:
        self._threshold = threshold
        self._inverse = inverse
        self.update()

    def _get_color(self) -> QColor:
        if self._inverse:
            return QColor("#FF3366") if self._value > self._threshold else QColor("#00FFcc")
        else:
            return QColor("#FF3366") if self._value < self._threshold else QColor("#00FFcc")

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2 - 10
        radius = min(w, h - 30) / 2 - 10
        pen_bg = QPen(QColor("#2D374B"))
        pen_bg.setWidth(8)
        pen_bg.setCapStyle(Qt.FlatCap)
        painter.setPen(pen_bg)
        painter.drawArc(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2), -30 * 16, 240 * 16)
        pen_marker = QPen(QColor("#A0AAB5"))
        pen_marker.setWidth(2)
        painter.setPen(pen_marker)
        angle = 210 - (self._threshold * 240)
        rad = math.radians(angle)
        mx1, my1 = cx + (radius - 12) * math.cos(rad), cy - (radius - 12) * math.sin(rad)
        mx2, my2 = cx + (radius + 12) * math.cos(rad), cy - (radius + 12) * math.sin(rad)
        painter.drawLine(QPointF(mx1, my1), QPointF(mx2, my2))
        pen_val = QPen(self._get_color())
        pen_val.setWidth(8)
        pen_val.setCapStyle(Qt.FlatCap)
        painter.setPen(pen_val)
        span_angle = int(self._value * 240 * -16)
        painter.drawArc(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2), 210 * 16, span_angle)
        painter.setPen(QColor("#A0AAB5"))
        font = QFont("Consolas", 11, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        painter.drawText(
            int(cx - fm.horizontalAdvance(self._title) / 2),
            int(h - 5),
            self._title,
        )
        val_str = f"{self._value:.2f}"
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Consolas", 16, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        painter.drawText(
            int(cx - fm.horizontalAdvance(val_str) / 2),
            int(cy + fm.height() / 4),
            val_str,
        )
        painter.end()


class AlertWidget(QWidget):
    """Displays the driver state banner (SAFE, WARNING, DISTRACTED)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.banner = QLabel("SAFE")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setObjectName("Banner_SAFE")
        self._layout.addWidget(self.banner)

    def set_state(self, state: str) -> None:
        self.banner.setText(state)
        self.banner.setObjectName(f"Banner_{state}")
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)
