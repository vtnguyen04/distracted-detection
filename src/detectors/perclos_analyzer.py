from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from src.domain.models import DetectionSignal, FaceLandmarks
from src.utils import batch_ear

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from src.config.settings import EarSettings, PerclosSettings


class PerclosAnalyzer:
    def __init__(self, perclos_settings: PerclosSettings, ear_settings: EarSettings, weight: float = 0.3) -> None:
        self._perclos_settings = perclos_settings
        self._ear_settings = ear_settings
        self._weight = weight
        self._history: deque[tuple[float, bool]] = deque()

    @property
    def name(self) -> str:
        return "perclos"

    @property
    def weight(self) -> float:
        return self._weight

    def detect(self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None) -> DetectionSignal:
        if landmarks is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        now = time.monotonic()
        left_eye = landmarks.get_pixel_coords(landmarks.left_eye_indices)
        right_eye = landmarks.get_pixel_coords(landmarks.right_eye_indices)
        ear = batch_ear(left_eye, right_eye)
        is_closed = ear < self._ear_settings.threshold
        self._history.append((now, is_closed))
        window_start = now - self._perclos_settings.window_sec
        while self._history and self._history[0][0] < window_start:
            self._history.popleft()
        if len(self._history) < 2:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.1)
        closed_count = sum((1 for _, closed in self._history if closed))
        perclos_value = closed_count / len(self._history)
        is_triggered = perclos_value > self._perclos_settings.threshold
        score = 0.0
        if is_triggered:
            excess = perclos_value - self._perclos_settings.threshold
            score = min(1.0, 0.5 + excess / (1.0 - self._perclos_settings.threshold) * 0.5)
        else:
            score = perclos_value / self._perclos_settings.threshold * 0.1
        window_fill = len(self._history) / max(1, self._perclos_settings.window_sec * 15)
        confidence = min(1.0, window_fill)
        return DetectionSignal(
            detector_name=self.name, value=perclos_value, score=score, is_triggered=is_triggered, confidence=confidence
        )

    def reset(self) -> None:
        self._history.clear()
