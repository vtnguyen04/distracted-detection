from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.domain.models import DetectionSignal, FaceLandmarks

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.config.settings import HeadPoseSettings
from src.config.constants import HEAD_POSE_MODEL_POINTS_3D

_MODEL_POINTS = np.array(HEAD_POSE_MODEL_POINTS_3D, dtype=np.float64)


def _rotation_matrix_to_euler(rotation_mat: np.ndarray) -> tuple[float, float, float]:
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    if pitch > 0:
        pitch = 180 - pitch
    else:
        pitch = -180 - pitch

    yaw = -yaw
    return pitch, yaw, roll


class HeadPoseDetector:
    def __init__(self, settings: HeadPoseSettings, weight: float = 0.15) -> None:
        self._settings = settings
        self._weight = weight
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    @property
    def name(self) -> str:
        return "head_pose"

    @property
    def weight(self) -> float:
        return self._weight

    def detect(self, landmarks: FaceLandmarks | None = None, frame: NDArray[np.uint8] | None = None) -> DetectionSignal:
        if landmarks is None:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        if self._camera_matrix is None:
            focal_length = landmarks.frame_width
            center = (landmarks.frame_width / 2, landmarks.frame_height / 2)
            self._camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
            )
        image_points = landmarks.get_pixel_coords(landmarks.face_oval_indices).astype(np.float64)
        success, rotation_vec, _translation_vec = cv2.solvePnP(
            _MODEL_POINTS, image_points, self._camera_matrix, self._dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            return DetectionSignal(detector_name=self.name, value=0.0, score=0.0, is_triggered=False, confidence=0.0)
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, _roll = _rotation_matrix_to_euler(rotation_mat)
        is_head_pitched = pitch > self._settings.pitch_threshold or pitch < -self._settings.pitch_threshold
        is_looking_away = abs(yaw) > self._settings.yaw_threshold
        is_triggered = is_head_pitched or is_looking_away
        score = 0.0
        if is_head_pitched:
            excess_pitch = abs(pitch) - self._settings.pitch_threshold
            score = min(1.0, 0.50 + excess_pitch / 25.0)

        if is_looking_away:
            excess_yaw = abs(yaw) - self._settings.yaw_threshold
            yaw_score = min(1.0, 0.50 + excess_yaw / 25.0)
            score = max(score, yaw_score)
        return DetectionSignal(detector_name=self.name, value=pitch, score=score, is_triggered=is_triggered)

    def reset(self) -> None:
        self._camera_matrix = None
