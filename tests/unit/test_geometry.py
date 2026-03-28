from __future__ import annotations

import numpy as np
from src.utils import batch_ear, euclidean_distance, eye_aspect_ratio, mouth_aspect_ratio


class TestGeometry:
    """Tests for EAR and MAR calculations."""

    def test_euclidean_distance_basic(self) -> None:
        """Basic distance calculation."""
        p1 = np.array([0, 0], dtype=np.float32)
        p2 = np.array([3, 4], dtype=np.float32)
        assert abs(euclidean_distance(p1, p2) - 5.0) < 0.001

    def test_euclidean_distance_same_point(self) -> None:
        """Distance between same point = 0."""
        p = np.array([5, 5], dtype=np.float32)
        assert euclidean_distance(p, p) == 0.0

    def test_ear_fully_open(self) -> None:
        """Wide-open eye should have high EAR."""
        eye = np.array(
            [
                [0, 50],
                [25, 20],
                [75, 20],
                [100, 50],
                [75, 80],
                [25, 80],
            ],
            dtype=np.float32,
        )
        ear = eye_aspect_ratio(eye)
        assert ear > 0.3

    def test_ear_fully_closed(self) -> None:
        """Closed eye should have very low EAR (~0)."""
        eye = np.array(
            [
                [0, 50],
                [25, 50],
                [75, 50],
                [100, 50],
                [75, 50],
                [25, 50],
            ],
            dtype=np.float32,
        )
        ear = eye_aspect_ratio(eye)
        assert ear < 0.05

    def test_ear_zero_horizontal(self) -> None:
        """Zero horizontal distance should return 0 (no division error)."""
        eye = np.array(
            [
                [50, 50],
                [50, 40],
                [50, 40],
                [50, 50],
                [50, 60],
                [50, 60],
            ],
            dtype=np.float32,
        )
        ear = eye_aspect_ratio(eye)
        assert ear == 0.0

    def test_mar_closed_mouth(self) -> None:
        """Closed mouth should have low MAR."""
        mouth = np.array(
            [
                [0, 50],
                [20, 48],
                [40, 48],
                [60, 48],
                [80, 50],
                [60, 52],
                [40, 52],
                [20, 52],
            ],
            dtype=np.float32,
        )
        mar = mouth_aspect_ratio(mouth)
        assert mar < 0.2

    def test_batch_ear_average(self) -> None:
        """batch_ear should average left and right EAR."""
        eye_open = np.array(
            [
                [0, 50],
                [25, 20],
                [75, 20],
                [100, 50],
                [75, 80],
                [25, 80],
            ],
            dtype=np.float32,
        )
        eye_closed = np.array(
            [
                [0, 50],
                [25, 50],
                [75, 50],
                [100, 50],
                [75, 50],
                [25, 50],
            ],
            dtype=np.float32,
        )
        avg = batch_ear(eye_open, eye_closed)
        individual_open = eye_aspect_ratio(eye_open)
        individual_closed = eye_aspect_ratio(eye_closed)
        expected = (individual_open + individual_closed) / 2.0
        assert abs(avg - expected) < 0.001
