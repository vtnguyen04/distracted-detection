from __future__ import annotations

from typing import TYPE_CHECKING

from src.detectors.ear_detector import EarDetector

if TYPE_CHECKING:
    from src.config.settings import EarSettings
    from src.domain.models import FaceLandmarks


class TestEarDetector:
    """Tests for Eye Aspect Ratio detection with consecutive frames."""

    def test_open_eyes_not_triggered(
        self,
        ear_settings: EarSettings,
        open_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Open eyes should produce EAR above threshold."""
        detector = EarDetector(settings=ear_settings)
        signal = detector.detect(open_eyes_landmarks)
        assert not signal.is_triggered
        assert signal.value > ear_settings.threshold
        assert signal.detector_name == "ear"

    def test_closed_eyes_single_frame_not_triggered(
        self,
        ear_settings: EarSettings,
        closed_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Single frame of closed eyes should NOT trigger (filters blinks)."""
        detector = EarDetector(settings=ear_settings)
        signal = detector.detect(closed_eyes_landmarks)
        assert signal.value < ear_settings.threshold
        assert not signal.is_triggered

    def test_closed_eyes_sustained_triggered(
        self,
        ear_settings: EarSettings,
        closed_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Sustained closed eyes (>= consecutive_frames) should trigger."""
        detector = EarDetector(settings=ear_settings)
        for _ in range(ear_settings.consecutive_frames):
            signal = detector.detect(closed_eyes_landmarks)
        assert signal.is_triggered
        assert signal.score >= 0.6
        assert signal.detector_name == "ear"

    def test_blink_resets_counter(
        self,
        ear_settings: EarSettings,
        closed_eyes_landmarks: FaceLandmarks,
        open_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Opening eyes resets the consecutive counter."""
        detector = EarDetector(settings=ear_settings)
        for _ in range(3):
            detector.detect(closed_eyes_landmarks)
        signal = detector.detect(open_eyes_landmarks)
        assert not signal.is_triggered
        signal = detector.detect(closed_eyes_landmarks)
        assert not signal.is_triggered

    def test_weight_configurable(self, ear_settings: EarSettings) -> None:
        """Detector weight should be configurable."""
        detector = EarDetector(settings=ear_settings, weight=0.5)
        assert detector.weight == 0.5

    def test_name_is_ear(self, ear_settings: EarSettings) -> None:
        """Detector name must be 'ear'."""
        detector = EarDetector(settings=ear_settings)
        assert detector.name == "ear"

    def test_reset_clears_counter(
        self,
        ear_settings: EarSettings,
        closed_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Reset should clear the consecutive counter."""
        detector = EarDetector(settings=ear_settings)
        for _ in range(3):
            detector.detect(closed_eyes_landmarks)
        detector.reset()
        signal = detector.detect(closed_eyes_landmarks)
        assert not signal.is_triggered
