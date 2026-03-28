from __future__ import annotations

from typing import TYPE_CHECKING

from src.detectors.mar_detector import MarDetector

if TYPE_CHECKING:
    from src.config.settings import MarSettings
    from src.domain.models import FaceLandmarks


class TestMarDetector:
    """Tests for Mouth Aspect Ratio detection."""

    def test_closed_mouth_not_triggered(
        self,
        mar_settings: MarSettings,
        open_eyes_landmarks: FaceLandmarks,
    ) -> None:
        """Closed mouth should not trigger yawning detection."""
        detector = MarDetector(settings=mar_settings)
        signal = detector.detect(open_eyes_landmarks)
        assert not signal.is_triggered
        assert signal.score == 0.0
        assert signal.detector_name == "mar"

    def test_yawning_triggered(
        self,
        mar_settings: MarSettings,
        yawning_landmarks: FaceLandmarks,
    ) -> None:
        """Wide open mouth should trigger yawning detection."""
        detector = MarDetector(settings=mar_settings)
        signal = detector.detect(yawning_landmarks)
        assert signal.is_triggered
        assert signal.value > mar_settings.threshold
        assert signal.score > 0.0
        assert signal.detector_name == "mar"

    def test_name_is_mar(self, mar_settings: MarSettings) -> None:
        """Detector name must be 'mar'."""
        detector = MarDetector(settings=mar_settings)
        assert detector.name == "mar"
