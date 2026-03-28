from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import structlog

from src.config.constants import (
    CLASS_NAME_TO_ID,
    DET_CONFIDENCE,
    DET_CONSECUTIVE_CLOSED,
    DET_CROP_X1,
    DET_CROP_X2,
    DET_CROP_Y1,
    DET_CROP_Y2,
    DET_EYE_CLOSED_COUNT,
    DET_EYE_OPEN_COUNT,
    DET_SCORE,
    DET_TRIGGERED,
    DET_VALUE,
    DETECTION_BUFFER_SIZE,
    DETECTION_HEADER_SIZE,
    DETECTION_STRIDE,
    MAX_DETECTIONS,
)
from src.engine.worker_base import WorkerProcess

logger = structlog.get_logger()


class InferenceWorker(WorkerProcess):
    """Runs YOLO .engine/.onnx/.xml inference on GPU/CPU, writes detections to SharedMemory."""

    def __init__(
        self,
        running: Any,
        settings: Any,
        frame_reader: Any,
        new_frame_event: Any,
        detection_shm_name: str,
        detections_ready_event: Any,
        landmarks_shm_name: str,
        landmarks_shape: tuple[int, ...],
    ) -> None:
        super().__init__(running, worker_name="inference_worker")
        self._settings = settings
        self._frame_reader = frame_reader
        self._new_frame_event = new_frame_event
        self._detection_shm_name = detection_shm_name
        self._detections_ready_event = detections_ready_event
        self._landmarks_shm_name = landmarks_shm_name
        self._landmarks_shape = landmarks_shape
        self._backend: Any = None

    def setup(self) -> None:
        from multiprocessing.shared_memory import SharedMemory

        from src.infrastructure.inference_backend import create_backend

        self._backend = create_backend(
            self._settings.inference.backend,
            self._settings.inference.distracted_model_path,
            device=self._settings.inference.device,
            imgsz=(self._settings.inference.crop_size, self._settings.inference.crop_size),
        )
        self._det_shm = SharedMemory(name=self._detection_shm_name, create=False)
        self._det_array = np.ndarray((DETECTION_BUFFER_SIZE,), dtype=np.float32, buffer=self._det_shm.buf)
        self._lm_shm = SharedMemory(name=self._landmarks_shm_name, create=False)
        self._lm_array = np.ndarray(self._landmarks_shape, dtype=np.float32, buffer=self._lm_shm.buf)
        self._consecutive_closed = 0
        self._ear_consecutive_frames = self._settings.ear.consecutive_frames
        self._confidence_threshold = self._settings.inference.confidence_threshold
        self._crop_size = self._settings.inference.crop_size

    def _get_face_crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop region centered on face using cached landmarks from MediaPipeWorker."""
        h, w = frame.shape[:2]
        size = self._crop_size
        face_indices = [1, 33, 263, 61, 291, 199]
        face_pts = self._lm_array[face_indices, :2]
        if np.any(face_pts > 0.01):
            cx = int(np.mean(face_pts[:, 0]) * w)
            cy = int(np.mean(face_pts[:, 1]) * h)
        else:
            cx, cy = w // 2, h // 2
        half = size // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= x2 - w
            x2 = w
        if y2 > h:
            y1 -= y2 - h
            y2 = h
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[:2] != (size, size):
            cropped = cv2.resize(cropped, (size, size))
        return cropped, (x1, y1, x2, y2)

    def process_frame(self) -> None:
        if not self._new_frame_event.wait(timeout=0.1):
            return
        frame = self._frame_reader.read()
        cropped, (fx1, fy1, fx2, fy2) = self._get_face_crop(frame)
        detections = self._backend.predict(cropped, confidence=self._confidence_threshold)
        eye_open_count = 0
        eye_closed_count = 0
        for det in detections:
            if det["class_name"] == "Eye closed":
                eye_closed_count += 1
            elif det["class_name"] == "Eye open":
                eye_open_count += 1
        is_eyes_closed = eye_closed_count > eye_open_count
        self._consecutive_closed = self._consecutive_closed + 1 if is_eyes_closed else 0
        is_distracted = self._consecutive_closed >= self._ear_consecutive_frames
        score = 0.0
        if is_distracted:
            duration_factor = min(2.0, self._consecutive_closed / self._ear_consecutive_frames)
            score = max(0.6, min(1.0, 0.6 * duration_factor))
        elif is_eyes_closed:
            score = 0.2 * (self._consecutive_closed / self._ear_consecutive_frames)
        total_eyes = eye_closed_count + eye_open_count
        self._det_array[DET_EYE_OPEN_COUNT] = float(eye_open_count)
        self._det_array[DET_EYE_CLOSED_COUNT] = float(eye_closed_count)
        self._det_array[DET_SCORE] = score
        self._det_array[DET_TRIGGERED] = 1.0 if is_distracted else 0.0
        self._det_array[DET_CONFIDENCE] = min(1.0, total_eyes / 2.0)
        self._det_array[DET_CONSECUTIVE_CLOSED] = float(self._consecutive_closed)
        self._det_array[DET_VALUE] = float(eye_closed_count) / max(total_eyes, 1)
        self._det_array[DET_CROP_X1] = float(fx1)
        self._det_array[DET_CROP_Y1] = float(fy1)
        self._det_array[DET_CROP_X2] = float(fx2)
        self._det_array[DET_CROP_Y2] = float(fy2)
        n_dets = min(len(detections), MAX_DETECTIONS)
        for i in range(n_dets):
            det = detections[i]
            offset = DETECTION_HEADER_SIZE + i * DETECTION_STRIDE
            bx1, by1, bx2, by2 = det["bbox"]
            cls_id = CLASS_NAME_TO_ID.get(det["class_name"], 9)
            self._det_array[offset] = float(bx1)
            self._det_array[offset + 1] = float(by1)
            self._det_array[offset + 2] = float(bx2)
            self._det_array[offset + 3] = float(by2)
            self._det_array[offset + 4] = float(cls_id)
            self._det_array[offset + 5] = float(det["confidence"])
        for i in range(n_dets, MAX_DETECTIONS):
            offset = DETECTION_HEADER_SIZE + i * DETECTION_STRIDE
            self._det_array[offset : offset + DETECTION_STRIDE] = 0.0
        self._detections_ready_event.set()

    def cleanup(self) -> None:
        if hasattr(self, "_backend") and self._backend:
            self._backend.release()
        if hasattr(self, "_det_shm"):
            self._det_shm.close()
        if hasattr(self, "_lm_shm"):
            self._lm_shm.close()
