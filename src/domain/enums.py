from __future__ import annotations

from enum import Enum, StrEnum, auto


class DriverState(Enum):
    ALERT = auto()
    WARNING = auto()
    DISTRACTED = auto()


class AlertLevel(Enum):
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class DetectorType(StrEnum):
    EAR = "ear"
    MAR = "mar"
    HEAD_POSE = "head_pose"
    PERCLOS = "perclos"
    BLINK_RATE = "blink_rate"
    YOLO_EYE = "yolo_eye"
