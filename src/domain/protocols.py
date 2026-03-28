from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from src.domain.enums import AlertLevel
    from src.domain.models import DetectionSignal, FaceLandmarks


@runtime_checkable
class Detector(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def weight(self) -> float: ...
    def detect(
        self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None
    ) -> DetectionSignal: ...
    def reset(self) -> None: ...


@runtime_checkable
class CameraProvider(Protocol):
    def read_frame(self) -> tuple[bool, NDArray[np.uint8]]: ...
    def release(self) -> None: ...
    @property
    def is_opened(self) -> bool: ...


@runtime_checkable
class FaceMeshProvider(Protocol):
    def get_landmarks(self, frame: NDArray[np.uint8]) -> FaceLandmarks | None: ...
    def release(self) -> None: ...


@runtime_checkable
class AlertProvider(Protocol):
    def play_alert(self, level: AlertLevel) -> None: ...
    def stop_alert(self) -> None: ...
    @property
    def is_playing(self) -> bool: ...


@runtime_checkable
class FrameRenderer(Protocol):
    def render(
        self,
        frame: NDArray[np.uint8],
        landmarks: FaceLandmarks | None,
        signals: list[DetectionSignal],
        distraction_score: float,
        driver_state_name: str,
        fps: float,
    ) -> NDArray[np.uint8]: ...
