from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def euclidean_distance(p1: NDArray[np.float32], p2: NDArray[np.float32]) -> float:
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(eye_points: NDArray[np.float32]) -> float:
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    h = euclidean_distance(eye_points[0], eye_points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def mouth_aspect_ratio(mouth_points: NDArray[np.float32]) -> float:
    v1 = euclidean_distance(mouth_points[1], mouth_points[7])
    v2 = euclidean_distance(mouth_points[2], mouth_points[6])
    v3 = euclidean_distance(mouth_points[3], mouth_points[5])
    h = euclidean_distance(mouth_points[0], mouth_points[4])
    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (2.0 * h)


def batch_ear(left_eye: NDArray[np.float32], right_eye: NDArray[np.float32]) -> float:
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0
