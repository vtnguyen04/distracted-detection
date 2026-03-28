from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.domain.models import DetectionSignal, FaceLandmarks
from src.utils import batch_ear

if TYPE_CHECKING:
    from src.config.settings import BlinkRateSettings, EarSettings


class BlinkRateDetector:
    def __init__(self, blink_settings: BlinkRateSettings, ear_settings: EarSettings, weight: float = 0.05) -> None:
        self._blink_settings = blink_settings
        self._ear_settings = ear_settings
        self._weight = weight
        self._blink_timestamps: deque[float] = deque()
        self._prev_closed = False
        self._closed_frames = 0
        self._start_time = time.monotonic()

    @property
    def name(self) -> str:
        return "blink_rate"

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
        if not hasattr(self, "_blink_start_time"):
            self._blink_start_time = now
            self._last_blink_duration = 0.0
        if not self._prev_closed and is_closed:
            self._blink_start_time = now
        if self._prev_closed and (not is_closed) and (self._closed_frames >= 1):
            self._blink_timestamps.append(now)
            self._last_blink_duration = (now - self._blink_start_time) * 1000.0
        if is_closed:
            self._closed_frames += 1
        else:
            self._closed_frames = 0
            if hasattr(self, "_last_blink_duration") and self._last_blink_duration > 600:
                self._last_blink_duration = 0.0
        self._prev_closed = is_closed
        window_start = now - self._blink_settings.window_sec
        while self._blink_timestamps and self._blink_timestamps[0] < window_start:
            self._blink_timestamps.popleft()
        elapsed = min(now - self._start_time, self._blink_settings.window_sec)
        blink_rate = len(self._blink_timestamps) * (60.0 / max(elapsed, 1.0)) if elapsed > 0 else 0.0
        is_abnormal_low = blink_rate < self._blink_settings.normal_low and elapsed >= 30.0
        is_abnormal_high_bpm = blink_rate > self._blink_settings.normal_high and elapsed >= 30.0
        is_microsleep = self._last_blink_duration > 600
        is_abnormal_high = is_microsleep or is_abnormal_high_bpm
        is_triggered = is_abnormal_low or is_abnormal_high
        score = 0.0
        if is_abnormal_low:
            score = 0.8
        elif is_abnormal_high:
            score = 0.7
            if self._last_blink_duration > 800:
                score = 1.0
        confidence = min(1.0, elapsed / 30.0)
        return DetectionSignal(
            detector_name=self.name,
            value=self._last_blink_duration,
            score=score,
            is_triggered=is_triggered,
            confidence=confidence,
        )

    def reset(self) -> None:
        self._blink_timestamps.clear()
        self._prev_closed = False
        self._closed_frames = 0
        self._start_time = time.monotonic()
