from __future__ import annotations

from typing import Any

import cv2
import structlog

from src.engine.worker_base import WorkerProcess
from src.infrastructure.camera import Camera

logger = structlog.get_logger()


class CaptureWorker(WorkerProcess):
    def __init__(self, running: Any, camera_settings: Any, frame_writer: Any, new_frame_event: Any) -> None:
        super().__init__(running, worker_name="capture_worker")
        self._camera_settings = camera_settings
        self._frame_writer = frame_writer
        self._new_frame_event = new_frame_event
        self._camera: Camera | None = None

    def setup(self) -> None:
        self._camera = Camera(self._camera_settings)

    def process_frame(self) -> None:
        if self._camera is None:
            return
        success, frame = self._camera.read_frame()
        if not success:
            self._running.value = 0
            return
        frame = cv2.resize(frame, (640, 480))
        self._frame_writer.write(frame)
        self._new_frame_event.clear()
        self._new_frame_event.set()

    def cleanup(self) -> None:
        if self._camera:
            self._camera.release()
