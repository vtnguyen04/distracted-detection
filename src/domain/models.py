from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.enums import AlertLevel, DriverState


@dataclass(frozen=True)
class FaceLandmarks:
    landmarks: NDArray[np.float32]
    frame_width: int
    frame_height: int

    @property
    def left_eye_indices(self) -> list[int]:
        return [362, 385, 387, 263, 373, 380]

    @property
    def right_eye_indices(self) -> list[int]:
        return [33, 160, 158, 133, 153, 144]

    @property
    def mouth_indices(self) -> list[int]:
        return [61, 39, 0, 269, 291, 405, 17, 181]

    @property
    def nose_tip_index(self) -> int:
        return 1

    @property
    def face_oval_indices(self) -> list[int]:
        return [1, 152, 263, 33, 291, 61]

    def get_pixel_coords(self, indices: list[int]) -> NDArray[np.float32]:
        points = self.landmarks[indices, :2]
        return np.column_stack([points[:, 0] * self.frame_width, points[:, 1] * self.frame_height]).astype(np.float32)


@dataclass(frozen=True)
class DetectionSignal:
    detector_name: str
    value: float
    score: float
    is_triggered: bool
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass()
class DetectionResult:
    signals: list[DetectionSignal] = field(default_factory=list)
    distraction_score: float = 0.0
    driver_state: DriverState | None = None
    alert_level: AlertLevel | None = None
    timestamp: float = field(default_factory=time.monotonic)
    fps: float = 0.0
    landmarks: FaceLandmarks | None = None
    face_detected: bool = False


@dataclass()
class DriverSnapshot:
    ear_history: list[float] = field(default_factory=list)
    mar_history: list[float] = field(default_factory=list)
    blink_timestamps: list[float] = field(default_factory=list)
    eye_closure_timestamps: list[tuple[float, bool]] = field(default_factory=list)
    head_pitch_history: list[float] = field(default_factory=list)
    window_start: float = field(default_factory=time.monotonic)

    def trim_to_window(self, window_sec: float) -> None:
        cutoff = time.monotonic() - window_sec
        self.ear_history = [v for v in self.ear_history if v > cutoff]
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff]
        self.eye_closure_timestamps = [(t, closed) for t, closed in self.eye_closure_timestamps if t > cutoff]
