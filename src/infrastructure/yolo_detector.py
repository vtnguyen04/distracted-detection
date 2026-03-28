from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import structlog

from src.domain.models import DetectionSignal, FaceLandmarks
from src.infrastructure.inference_backend import InferenceBackend, create_backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.config.settings import EarSettings, InferenceSettings
logger = structlog.get_logger()


class YoloEyeDetector:
    def __init__(self, inference_settings: InferenceSettings, ear_settings: EarSettings, weight: float = 0.35) -> None:
        self._settings = inference_settings
        self._ear_settings = ear_settings
        self._weight = weight
        self._consecutive_closed = 0
        self._backend: InferenceBackend | None = None
        self._face_bbox: tuple[int, int, int, int] | None = None
        self._last_face_detect_time = 0.0
        self._crop_size = (320, 320)
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            self._backend = create_backend(
                self._settings.backend,
                self._settings.distracted_model_path,
                device=self._settings.device,
                imgsz=self._crop_size,
            )
            logger.info(
                "yolo_eye_detector_initialized",
                backend=self._settings.backend,
                model=getattr(self._backend, "_loaded_path", self._settings.distracted_model_path),
            )
        except (ImportError, FileNotFoundError, Exception) as e:
            logger.warning("yolo_backend_unavailable", error=str(e))
            self._backend = None

    @property
    def name(self) -> str:
        return "yolo_eye"

    @property
    def weight(self) -> float:
        return self._weight

    def detect(self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None) -> DetectionSignal:
        if landmarks is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        if self._backend is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        face_indices = landmarks.face_oval_indices
        points = landmarks.get_pixel_coords(face_indices)
        x_min = max(0, int(np.min(points[:, 0])) - 40)
        y_min = max(0, int(np.min(points[:, 1])) - 40)
        x_max = min(landmarks.frame_width, int(np.max(points[:, 0])) + 40)
        y_max = min(landmarks.frame_height, int(np.max(points[:, 1])) + 40)
        self._face_bbox = (x_min, y_min, x_max, y_max)
        return DetectionSignal(
            detector_name=self.name,
            value=0.0,
            score=0.0,
            is_triggered=False,
            confidence=0.5,
            metadata={"face_bbox": self._face_bbox},
        )

    def detect_from_frame(self, frame: NDArray[np.uint8], landmarks: FaceLandmarks | None = None) -> DetectionSignal:
        if self._backend is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        if landmarks is not None:
            face_indices = landmarks.face_oval_indices
            points = landmarks.get_pixel_coords(face_indices)
            cx = int(np.mean(points[:, 0]))
            cy = int(np.mean(points[:, 1]))
        else:
            h, w = frame.shape[:2]
            cx, cy = (w // 2, h // 2)
        size_w, size_h = self._crop_size
        half_w, half_h = (size_w // 2, size_h // 2)
        h, w = frame.shape[:2]
        x_min = cx - half_w
        y_min = cy - half_h
        x_max = cx + half_w
        y_max = cy + half_h
        if x_min < 0:
            x_max -= x_min
            x_min = 0
        if y_min < 0:
            y_max -= y_min
            y_min = 0
        if x_max > w:
            x_min -= x_max - w
            x_max = w
        if y_max > h:
            y_min -= y_max - h
            y_max = h
        x_min, y_min = (max(0, x_min), max(0, y_min))
        x_max, y_max = (min(w, x_max), min(h, y_max))
        cropped = frame[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, self._crop_size) if cropped.shape[:2] != (size_h, size_w) else cropped
        detections = self._backend.predict(resized, confidence=self._settings.confidence_threshold)
        eye_closed_count = 0
        eye_open_count = 0
        for det in detections:
            if det["class_name"] == "Eye closed":
                eye_closed_count += 1
            elif det["class_name"] == "Eye open":
                eye_open_count += 1
        is_eyes_closed = eye_closed_count > eye_open_count
        if is_eyes_closed:
            self._consecutive_closed += 1
        else:
            self._consecutive_closed = 0
        is_distracted = self._consecutive_closed >= self._ear_settings.consecutive_frames
        score = 0.0
        if is_distracted:
            duration_factor = min(2.0, self._consecutive_closed / self._ear_settings.consecutive_frames)
            score = min(1.0, 0.6 * duration_factor)
            score = max(0.6, score)
        elif is_eyes_closed:
            score = 0.2 * (self._consecutive_closed / self._ear_settings.consecutive_frames)
        total_eyes = eye_closed_count + eye_open_count
        value = eye_closed_count / max(total_eyes, 1)
        return DetectionSignal(
            detector_name=self.name,
            value=value,
            score=score,
            is_triggered=is_distracted,
            confidence=min(1.0, total_eyes / 2.0),
            metadata={"detections": detections, "face_bbox": (x_min, y_min, x_max, y_max)},
        )

    def reset(self) -> None:
        self._consecutive_closed = 0
        self._face_bbox = None

    def release(self) -> None:
        if self._backend:
            self._backend.release()
