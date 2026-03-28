from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.domain.models import DetectionSignal, FaceLandmarks
from src.utils import batch_ear

if TYPE_CHECKING:
    from src.config.settings import EarSettings


class EarDetector:
    def __init__(self, settings: EarSettings, weight: float = 0.35) -> None:
        self._settings = settings
        self._weight = weight
        self._consecutive_closed = 0

    @property
    def name(self) -> str:
        return "ear"

    @property
    def weight(self) -> float:
        return self._weight

    def detect(self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None) -> DetectionSignal:
        if landmarks is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        left_eye = landmarks.get_pixel_coords(landmarks.left_eye_indices)
        right_eye = landmarks.get_pixel_coords(landmarks.right_eye_indices)
        ear_value = batch_ear(left_eye, right_eye)
        is_below = ear_value < self._settings.threshold
        if is_below:
            self._consecutive_closed += 1
        else:
            self._consecutive_closed = 0
        is_distracted_closure = self._consecutive_closed >= self._settings.consecutive_frames
        score = 0.0
        if is_distracted_closure:
            distance_ratio = max(0.0, 1.0 - ear_value / self._settings.threshold)
            duration_factor = min(2.0, self._consecutive_closed / self._settings.consecutive_frames)
            score = min(1.0, distance_ratio * duration_factor)
            score = max(0.6, score)
        elif is_below:
            score = 0.2 * (self._consecutive_closed / self._settings.consecutive_frames)
        return DetectionSignal(
            detector_name=self.name, value=ear_value, score=score, is_triggered=is_distracted_closure
        )

    def reset(self) -> None:
        self._consecutive_closed = 0
