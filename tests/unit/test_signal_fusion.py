from __future__ import annotations

from src.detectors.signal_fusion import SignalFusion
from src.domain.models import DetectionSignal


class TestSignalFusion:
    """Tests for hybrid MAX + weighted signal fusion."""

    def test_empty_signals_returns_zero(self) -> None:
        """No signals should return 0.0."""
        fusion = SignalFusion()
        assert fusion.fuse([], {}) == 0.0

    def test_single_strong_signal_not_diluted(self) -> None:
        """A single strongly triggered signal should NOT be diluted to near-zero."""
        fusion = SignalFusion()
        signals = [
            DetectionSignal(
                detector_name="ear",
                value=0.15,
                score=0.8,
                is_triggered=True,
                confidence=1.0,
            ),
            DetectionSignal(
                detector_name="mar",
                value=0.3,
                score=0.0,
                is_triggered=False,
                confidence=1.0,
            ),
            DetectionSignal(
                detector_name="head_pose",
                value=0.0,
                score=0.0,
                is_triggered=False,
                confidence=1.0,
            ),
        ]
        weights = {"ear": 0.40, "mar": 0.10, "head_pose": 0.15}
        result = fusion.fuse(signals, weights)
        assert result >= 0.5, f"Strong signal diluted to {result}"

    def test_multiple_signals_compound(self) -> None:
        """Multiple triggered signals should compound for higher score."""
        fusion = SignalFusion()
        signals = [
            DetectionSignal(
                detector_name="ear",
                value=0.15,
                score=0.8,
                is_triggered=True,
                confidence=1.0,
            ),
            DetectionSignal(
                detector_name="perclos",
                value=0.3,
                score=0.7,
                is_triggered=True,
                confidence=1.0,
            ),
        ]
        weights = {"ear": 0.40, "perclos": 0.30}
        result = fusion.fuse(signals, weights)
        assert result >= 0.7

    def test_no_triggered_signals(self) -> None:
        """All-green signals should produce low score."""
        fusion = SignalFusion()
        signals = [
            DetectionSignal(
                detector_name="ear",
                value=0.35,
                score=0.1,
                is_triggered=False,
                confidence=1.0,
            ),
            DetectionSignal(
                detector_name="mar",
                value=0.3,
                score=0.0,
                is_triggered=False,
                confidence=1.0,
            ),
        ]
        result = fusion.fuse(signals, {"ear": 0.4, "mar": 0.1})
        assert result < 0.3

    def test_result_bounded_0_to_1(self) -> None:
        """Result should always be in [0.0, 1.0]."""
        fusion = SignalFusion()
        signals = [
            DetectionSignal(
                detector_name="ear",
                value=0.0,
                score=1.0,
                is_triggered=True,
                confidence=1.0,
            ),
            DetectionSignal(
                detector_name="mar",
                value=0.0,
                score=1.0,
                is_triggered=True,
                confidence=1.0,
            ),
        ]
        result = fusion.fuse(signals, {"ear": 0.5, "mar": 0.5})
        assert 0.0 <= result <= 1.0

    def test_unknown_detector_low_impact(self) -> None:
        """Signal from unregistered detector should have minimal impact."""
        fusion = SignalFusion()
        signals = [
            DetectionSignal(
                detector_name="unknown",
                value=0.5,
                score=1.0,
                is_triggered=True,
                confidence=1.0,
            ),
        ]
        result = fusion.fuse(signals, {"ear": 1.0})
        assert result <= 0.5
