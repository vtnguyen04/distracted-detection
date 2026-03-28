from __future__ import annotations

import numpy as np
import pytest
from src.config.settings import (
    BlinkRateSettings,
    EarSettings,
    HeadPoseSettings,
    MarSettings,
    PerclosSettings,
    StateMachineSettings,
)
from src.domain.models import FaceLandmarks


@pytest.fixture
def ear_settings() -> EarSettings:
    """Default EAR settings for testing."""
    return EarSettings(threshold=0.26, consecutive_frames=5)


@pytest.fixture
def mar_settings() -> MarSettings:
    """Default MAR settings for testing."""
    return MarSettings(threshold=0.6, min_duration_sec=1.5)


@pytest.fixture
def head_pose_settings() -> HeadPoseSettings:
    """Default head pose settings for testing."""
    return HeadPoseSettings(pitch_threshold=-15.0, yaw_threshold=30.0)


@pytest.fixture
def perclos_settings() -> PerclosSettings:
    """Default PERCLOS settings for testing."""
    return PerclosSettings(window_sec=60.0, threshold=0.15)


@pytest.fixture
def blink_settings() -> BlinkRateSettings:
    """Default blink rate settings for testing."""
    return BlinkRateSettings(normal_low=12, normal_high=20, window_sec=60.0)


@pytest.fixture
def state_machine_settings() -> StateMachineSettings:
    """Default state machine settings for testing."""
    return StateMachineSettings()


@pytest.fixture
def open_eyes_landmarks() -> FaceLandmarks:
    """Synthetic landmarks simulating wide-open eyes (high EAR)."""
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[33] = [0.30, 0.35, 0.0]
    landmarks[160] = [0.32, 0.32, 0.0]
    landmarks[158] = [0.35, 0.32, 0.0]
    landmarks[133] = [0.37, 0.35, 0.0]
    landmarks[153] = [0.35, 0.38, 0.0]
    landmarks[144] = [0.32, 0.38, 0.0]
    landmarks[362] = [0.63, 0.35, 0.0]
    landmarks[385] = [0.65, 0.32, 0.0]
    landmarks[387] = [0.68, 0.32, 0.0]
    landmarks[263] = [0.70, 0.35, 0.0]
    landmarks[373] = [0.68, 0.38, 0.0]
    landmarks[380] = [0.65, 0.38, 0.0]
    landmarks[61] = [0.42, 0.65, 0.0]
    landmarks[291] = [0.58, 0.65, 0.0]
    landmarks[39] = [0.44, 0.648, 0.0]
    landmarks[181] = [0.44, 0.652, 0.0]
    landmarks[0] = [0.50, 0.647, 0.0]
    landmarks[17] = [0.50, 0.653, 0.0]
    landmarks[269] = [0.56, 0.648, 0.0]
    landmarks[405] = [0.56, 0.652, 0.0]
    landmarks[1] = [0.50, 0.50, 0.0]
    landmarks[199] = [0.50, 0.80, 0.0]
    return FaceLandmarks(landmarks=landmarks, frame_width=640, frame_height=480)


@pytest.fixture
def closed_eyes_landmarks() -> FaceLandmarks:
    """Synthetic landmarks simulating closed eyes (low EAR)."""
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[33] = [0.30, 0.35, 0.0]
    landmarks[160] = [0.32, 0.349, 0.0]
    landmarks[158] = [0.35, 0.349, 0.0]
    landmarks[133] = [0.37, 0.35, 0.0]
    landmarks[153] = [0.35, 0.351, 0.0]
    landmarks[144] = [0.32, 0.351, 0.0]
    landmarks[362] = [0.63, 0.35, 0.0]
    landmarks[385] = [0.65, 0.349, 0.0]
    landmarks[387] = [0.68, 0.349, 0.0]
    landmarks[263] = [0.70, 0.35, 0.0]
    landmarks[373] = [0.68, 0.351, 0.0]
    landmarks[380] = [0.65, 0.351, 0.0]
    landmarks[61] = [0.42, 0.65, 0.0]
    landmarks[291] = [0.58, 0.65, 0.0]
    landmarks[39] = [0.44, 0.64, 0.0]
    landmarks[181] = [0.44, 0.66, 0.0]
    landmarks[0] = [0.50, 0.63, 0.0]
    landmarks[17] = [0.50, 0.67, 0.0]
    landmarks[269] = [0.56, 0.64, 0.0]
    landmarks[405] = [0.56, 0.66, 0.0]
    landmarks[1] = [0.50, 0.50, 0.0]
    landmarks[199] = [0.50, 0.80, 0.0]
    return FaceLandmarks(landmarks=landmarks, frame_width=640, frame_height=480)


@pytest.fixture
def yawning_landmarks() -> FaceLandmarks:
    """Synthetic landmarks simulating yawning (high MAR)."""
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[33] = [0.30, 0.35, 0.0]
    landmarks[160] = [0.32, 0.32, 0.0]
    landmarks[158] = [0.35, 0.32, 0.0]
    landmarks[133] = [0.37, 0.35, 0.0]
    landmarks[153] = [0.35, 0.38, 0.0]
    landmarks[144] = [0.32, 0.38, 0.0]
    landmarks[362] = [0.63, 0.35, 0.0]
    landmarks[385] = [0.65, 0.32, 0.0]
    landmarks[387] = [0.68, 0.32, 0.0]
    landmarks[263] = [0.70, 0.35, 0.0]
    landmarks[373] = [0.68, 0.38, 0.0]
    landmarks[380] = [0.65, 0.38, 0.0]
    landmarks[61] = [0.40, 0.65, 0.0]
    landmarks[291] = [0.60, 0.65, 0.0]
    landmarks[39] = [0.44, 0.58, 0.0]
    landmarks[181] = [0.44, 0.72, 0.0]
    landmarks[0] = [0.50, 0.56, 0.0]
    landmarks[17] = [0.50, 0.74, 0.0]
    landmarks[269] = [0.56, 0.58, 0.0]
    landmarks[405] = [0.56, 0.72, 0.0]
    landmarks[1] = [0.50, 0.50, 0.0]
    landmarks[199] = [0.50, 0.80, 0.0]
    return FaceLandmarks(landmarks=landmarks, frame_width=640, frame_height=480)
