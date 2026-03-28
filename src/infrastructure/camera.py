from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.config.settings import CameraSettings
logger = structlog.get_logger()


class Camera:
    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._cap: cv2.VideoCapture | None = None
        self._connect()

    def _connect(self) -> None:
        source = self._settings.source or self._settings.index
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        for attempt in range(1, self._settings.retry_attempts + 1):
            self._cap = cv2.VideoCapture(source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
                self._cap.set(cv2.CAP_PROP_FPS, self._settings.fps)
                logger.info(
                    "camera_connected",
                    source=source,
                    width=self._settings.width,
                    height=self._settings.height,
                    attempt=attempt,
                )
                return
            logger.warning(
                "camera_connect_retry", source=source, attempt=attempt, max_attempts=self._settings.retry_attempts
            )
            time.sleep(self._settings.retry_delay_sec)
        msg = f"Cannot open camera source: {source} after {self._settings.retry_attempts} attempts"
        raise RuntimeError(msg)

    def read_frame(self) -> tuple[bool, NDArray[np.uint8]]:
        if self._cap is None or not self._cap.isOpened():
            return (False, np.empty(0, dtype=np.uint8))
        ret, frame = self._cap.read()
        if not ret:
            return (False, np.empty(0, dtype=np.uint8))
        return (True, frame)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("camera_released")

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
