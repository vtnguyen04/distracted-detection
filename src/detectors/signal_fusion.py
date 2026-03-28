from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.models import DetectionSignal


class SignalFusion:
    def fuse(self, signals: list[DetectionSignal], weights: dict[str, float]) -> float:
        if not signals:
            return 0.0
        total_weighted = 0.0
        total_weight = 0.0
        for signal in signals:
            w = weights.get(signal.detector_name, 0.0)
            total_weighted += signal.score * w * signal.confidence
            total_weight += w * signal.confidence
        avg_score = total_weighted / total_weight if total_weight > 0 else 0.0
        max_triggered = 0.0
        for signal in signals:
            if signal.is_triggered and signal.score >= 0.6:
                w = weights.get(signal.detector_name, 0.0)
                if w > 0:
                    score = signal.score
                    if signal.detector_name == "head_pose":
                        score *= 0.7
                    max_triggered = max(max_triggered, score)
        fused = max(avg_score, max_triggered)
        return min(1.0, max(0.0, fused))
