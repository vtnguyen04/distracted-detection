from __future__ import annotations

from typing import TYPE_CHECKING

from src.detectors.blink_rate_detector import BlinkRateDetector
from src.detectors.ear_detector import EarDetector
from src.detectors.head_pose_detector import HeadPoseDetector
from src.detectors.mar_detector import MarDetector
from src.detectors.perclos_analyzer import PerclosAnalyzer
from src.domain.enums import DetectorType

if TYPE_CHECKING:
    from src.config.settings import AppSettings
    from src.domain.protocols import Detector


class DetectorFactory:
    @staticmethod
    def create_all(settings: AppSettings, exclude: set[str] | None = None) -> list[Detector]:
        enabled = settings.pipeline.enabled_detectors
        weights = settings.weights
        exclude = exclude or set()
        detectors: list[Detector] = []
        registry: dict[str, Detector] = {
            DetectorType.EAR: EarDetector(settings=settings.ear, weight=weights.ear),
            DetectorType.MAR: MarDetector(settings=settings.mar, weight=weights.mar),
            DetectorType.HEAD_POSE: HeadPoseDetector(settings=settings.head_pose, weight=weights.head_pose),
            DetectorType.PERCLOS: PerclosAnalyzer(
                perclos_settings=settings.perclos, ear_settings=settings.ear, weight=weights.perclos
            ),
            DetectorType.BLINK_RATE: BlinkRateDetector(
                blink_settings=settings.blink_rate, ear_settings=settings.ear, weight=weights.blink_rate
            ),
        }
        if "yolo_eye" in enabled and "yolo_eye" not in exclude:
            from src.infrastructure.yolo_detector import YoloEyeDetector

            registry["yolo_eye"] = YoloEyeDetector(
                inference_settings=settings.inference, ear_settings=settings.ear, weight=weights.yolo_eye
            )
        for detector_name in enabled:
            if detector_name in exclude:
                continue
            detector = registry.get(detector_name)
            if detector is not None:
                detectors.append(detector)
        return detectors
