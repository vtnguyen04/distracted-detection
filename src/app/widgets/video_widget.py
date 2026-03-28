import numpy as np
from PySide6.QtCore import QRectF, Qt, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Displays the annotated camera frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self._current_pixmap = None

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the displayed image from an OpenCV BGR frame."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = frame[..., ::-1].copy()
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qt_img)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Cyberpunk curved background/borders
        rect = QRectF(self.rect())
        radius = 24.0

        # Draw background canvas
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.fillPath(path, QColor("#0B0E14"))

        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            # Clip pixel map to the smooth rounded path bounds
            painter.setClipPath(path)

            scaled = self._current_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = int((self.width() - scaled.width()) / 2)
            y = int((self.height() - scaled.height()) / 2)
            painter.drawPixmap(x, y, scaled)

            # Release clipping for border overlay
            painter.setClipping(False)

        # Apply glowing accent border
        pen = QPen(QColor(0, 255, 204, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, radius, radius)
