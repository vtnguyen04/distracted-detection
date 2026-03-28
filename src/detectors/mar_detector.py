from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.models import DetectionSignal, FaceLandmarks
from src.utils import mouth_aspect_ratio

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from src.config.settings import MarSettings


class MarDetector:
    def __init__(self, settings: MarSettings, weight: float = 0.15) -> None:
        self._settings = settings
        self._weight = weight

    @property
    def name(self) -> str:
        return "mar"

    @property
    def weight(self) -> float:
        return self._weight

    def detect(self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None) -> DetectionSignal:
        if landmarks is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        mouth_points = landmarks.get_pixel_coords(landmarks.mouth_indices)
        mar_value = mouth_aspect_ratio(mouth_points)
        is_yawning = mar_value > self._settings.threshold
        score = 0.0
        if is_yawning:
            excess = mar_value - self._settings.threshold
            score = min(1.0, excess / self._settings.threshold)
        return DetectionSignal(detector_name=self.name, value=mar_value, score=score, is_triggered=is_yawning)

    def reset(self) -> None:
        """Reset the detector state."""
        pass
