"""Centralized constants for the distracted detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LandmarkIndex:
    """MediaPipe 478-point face landmark indices for specific features."""

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 39, 0, 269, 291, 405, 17, 181]
    NOSE_TIP = 1
    FACE_OVAL = [1, 152, 263, 33, 454, 234]

    LANDMARKS_SHAPE: tuple[int, int] = (478, 3)


class LandmarkPath:
    """Ordered index sequences for drawing face mesh contours."""

    # fmt: off
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109, 10,
    ]
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    LIPS_OUTER = [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 409, 270, 269, 267, 0,
        37, 39, 40, 185, 61,
    ]
    LIPS_INNER = [
        78, 191, 80, 81, 82, 13, 312, 311,
        310, 415, 308, 324, 318, 402, 317, 14,
        87, 178, 88, 95, 78,
    ]
    # fmt: on


class SignalBuffer:
    """Shared-memory layout for landmark-detector signal exchange.

    Buffer: [det0_val, det0_score, det0_triggered, det1_..., ..., metadata_slots]
    """

    SLOTS_PER_DETECTOR = 3
    MAX_DETECTORS = 5
    METADATA_SLOTS = 13
    SIZE = MAX_DETECTORS * SLOTS_PER_DETECTOR + METADATA_SLOTS


class SignalMetaOffset(IntEnum):
    """Negative indices into the signal buffer for metadata fields."""

    FACE_DETECTED = -13
    FRAME_WIDTH = -12
    FRAME_HEIGHT = -11
    FPS = -10
    PITCH = -9
    YAW = -8
    ROLL = -7
    RVEC_0 = -6
    RVEC_1 = -5
    RVEC_2 = -4
    TVEC_0 = -3
    TVEC_1 = -2
    TVEC_2 = -1


class DetectionBuffer:
    """Shared-memory layout for YOLO detection results.

    Buffer: [header(11 floats)] + [det0(6 floats), det1(6), ...]
    """

    MAX_DETECTIONS = 10
    STRIDE = 6
    HEADER_SIZE = 11
    SIZE = HEADER_SIZE + STRIDE * MAX_DETECTIONS


class DetectionHeaderOffset(IntEnum):
    """Offsets within the detection buffer header."""

    EYE_OPEN_COUNT = 0
    EYE_CLOSED_COUNT = 1
    SCORE = 2
    TRIGGERED = 3
    CONFIDENCE = 4
    CONSECUTIVE_CLOSED = 5
    VALUE = 6
    CROP_X1 = 7
    CROP_Y1 = 8
    CROP_X2 = 9
    CROP_Y2 = 10


class YoloClass:
    """YOLO model class ID ↔ name mappings."""

    FULL_NAMES: dict[int, str] = {
        0: "Face",
        1: "Eye open",
        2: "Eye closed",
        3: "Eye open",
        4: "Eye closed",
        5: "Mouth",
    }

    NAME_TO_ID: dict[str, int] = {
        "Eye open": 0,
        "Eye closed": 1,
        "Mouth": 2,
        "Face": 3,
    }

    ID_TO_NAME: dict[int, str] = {
        0: "Eye open",
        1: "Eye closed",
        2: "Mouth",
        3: "Face",
    }


class DetectorRegistry:
    """Canonical detector names and ordering."""

    LANDMARK_NAMES = ["ear", "mar", "head_pose", "perclos", "blink_rate"]
    YOLO_NAME = "yolo_eye"


@dataclass(frozen=True)
class HeadPoseModelPoint:
    """3D reference points for cv2.solvePnP head pose estimation."""

    POINTS = [
        (0.0, 0.0, 0.0),
        (0.0, 330.0, -65.0),
        (-225.0, -170.0, -135.0),
        (225.0, -170.0, -135.0),
        (-150.0, 150.0, -125.0),
        (150.0, 150.0, -125.0),
    ]


@dataclass(frozen=True)
class Color:
    """OpenCV BGR color palette."""

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    DARK_BG = (15, 10, 10)

    MESH_FACE = (255, 255, 0)
    MESH_EYE = (0, 255, 255)
    MESH_LIP = (0, 255, 0)
    MESH_BROW = (0, 165, 255)

    RETICLE = (0, 165, 255)
    AXIS_X = (0, 0, 255)
    AXIS_Y = (0, 255, 0)
    AXIS_Z = (255, 0, 0)


# Backward-compatible aliases (used by existing imports across the codebase)
LANDMARKS_SHAPE = LandmarkIndex.LANDMARKS_SHAPE
SIGNAL_SLOTS_PER_DETECTOR = SignalBuffer.SLOTS_PER_DETECTOR
MAX_LANDMARK_DETECTORS = SignalBuffer.MAX_DETECTORS
SIGNAL_METADATA_SLOTS = SignalBuffer.METADATA_SLOTS
SIGNAL_BUFFER_SIZE = SignalBuffer.SIZE
SIG_FACE_DETECTED = SignalMetaOffset.FACE_DETECTED
SIG_FRAME_WIDTH = SignalMetaOffset.FRAME_WIDTH
SIG_FRAME_HEIGHT = SignalMetaOffset.FRAME_HEIGHT
SIG_FPS = SignalMetaOffset.FPS
SIG_PITCH = SignalMetaOffset.PITCH
SIG_YAW = SignalMetaOffset.YAW
SIG_ROLL = SignalMetaOffset.ROLL
SIG_RVEC_0 = SignalMetaOffset.RVEC_0
SIG_RVEC_1 = SignalMetaOffset.RVEC_1
SIG_RVEC_2 = SignalMetaOffset.RVEC_2
SIG_TVEC_0 = SignalMetaOffset.TVEC_0
SIG_TVEC_1 = SignalMetaOffset.TVEC_1
SIG_TVEC_2 = SignalMetaOffset.TVEC_2
DETECTION_HEADER_SIZE = DetectionBuffer.HEADER_SIZE
DETECTION_STRIDE = DetectionBuffer.STRIDE
MAX_DETECTIONS = DetectionBuffer.MAX_DETECTIONS
DETECTION_BUFFER_SIZE = DetectionBuffer.SIZE
DET_EYE_OPEN_COUNT = DetectionHeaderOffset.EYE_OPEN_COUNT
DET_EYE_CLOSED_COUNT = DetectionHeaderOffset.EYE_CLOSED_COUNT
DET_SCORE = DetectionHeaderOffset.SCORE
DET_TRIGGERED = DetectionHeaderOffset.TRIGGERED
DET_CONFIDENCE = DetectionHeaderOffset.CONFIDENCE
DET_CONSECUTIVE_CLOSED = DetectionHeaderOffset.CONSECUTIVE_CLOSED
DET_VALUE = DetectionHeaderOffset.VALUE
DET_CROP_X1 = DetectionHeaderOffset.CROP_X1
DET_CROP_Y1 = DetectionHeaderOffset.CROP_Y1
DET_CROP_X2 = DetectionHeaderOffset.CROP_X2
DET_CROP_Y2 = DetectionHeaderOffset.CROP_Y2
YOLO_CLASS_NAMES = YoloClass.FULL_NAMES
CLASS_NAME_TO_ID = YoloClass.NAME_TO_ID
ID_TO_CLASS_NAME = YoloClass.ID_TO_NAME
LEFT_EYE_INDICES = LandmarkIndex.LEFT_EYE
RIGHT_EYE_INDICES = LandmarkIndex.RIGHT_EYE
MOUTH_INDICES = LandmarkIndex.MOUTH
NOSE_TIP_INDEX = LandmarkIndex.NOSE_TIP
FACE_OVAL_INDICES = LandmarkIndex.FACE_OVAL
FACE_OVAL_PATH = LandmarkPath.FACE_OVAL
LEFT_EYEBROW_PATH = LandmarkPath.LEFT_EYEBROW
RIGHT_EYEBROW_PATH = LandmarkPath.RIGHT_EYEBROW
LEFT_EYE_PATH = LandmarkPath.LEFT_EYE
RIGHT_EYE_PATH = LandmarkPath.RIGHT_EYE
LIPS_OUTER_PATH = LandmarkPath.LIPS_OUTER
LIPS_INNER_PATH = LandmarkPath.LIPS_INNER
COLOR_GREEN = Color.GREEN
COLOR_YELLOW = Color.YELLOW
COLOR_RED = Color.RED
COLOR_WHITE = Color.WHITE
COLOR_CYAN = Color.CYAN
COLOR_MAGENTA = Color.MAGENTA
COLOR_DARK_BG = Color.DARK_BG
COLOR_MESH_FACE = Color.MESH_FACE
COLOR_MESH_EYE = Color.MESH_EYE
COLOR_MESH_LIP = Color.MESH_LIP
COLOR_MESH_BROW = Color.MESH_BROW
COLOR_RETICLE = Color.RETICLE
COLOR_AXIS_X = Color.AXIS_X
COLOR_AXIS_Y = Color.AXIS_Y
COLOR_AXIS_Z = Color.AXIS_Z
HEAD_POSE_MODEL_POINTS_3D = HeadPoseModelPoint.POINTS
LANDMARK_DETECTOR_NAMES = DetectorRegistry.LANDMARK_NAMES
YOLO_DETECTOR_NAME = DetectorRegistry.YOLO_NAME
