from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import structlog

from src.config.constants import (
    HEAD_POSE_MODEL_POINTS_3D,
    SIG_FACE_DETECTED,
    SIG_FPS,
    SIG_FRAME_HEIGHT,
    SIG_FRAME_WIDTH,
    SIG_PITCH,
    SIG_ROLL,
    SIG_RVEC_0,
    SIG_RVEC_1,
    SIG_RVEC_2,
    SIG_TVEC_0,
    SIG_TVEC_1,
    SIG_TVEC_2,
    SIG_YAW,
    SIGNAL_BUFFER_SIZE,
    SIGNAL_SLOTS_PER_DETECTOR,
)
from src.engine.worker_base import WorkerProcess

logger = structlog.get_logger()
_MODEL_POINTS_3D = np.array(HEAD_POSE_MODEL_POINTS_3D, dtype=np.float64)


class MediaPipeWorker(WorkerProcess):
    """Runs MediaPipe face landmarks + lightweight detectors (EAR, MAR, HeadPose, etc.)."""

    def __init__(
        self,
        running: Any,
        settings: Any,
        frame_reader: Any,
        new_frame_event: Any,
        landmarks_shm_name: str,
        landmarks_shape: tuple[int, ...],
        signal_shm_name: str,
        landmarks_ready_event: Any,
    ) -> None:
        super().__init__(running, worker_name="mediapipe_worker")
        self._settings = settings
        self._frame_reader = frame_reader
        self._new_frame_event = new_frame_event
        self._landmarks_shm_name = landmarks_shm_name
        self._landmarks_shape = landmarks_shape
        self._signal_shm_name = signal_shm_name
        self._landmarks_ready_event = landmarks_ready_event

    def setup(self) -> None:
        from multiprocessing.shared_memory import SharedMemory

        from src.detectors.factory import DetectorFactory
        from src.infrastructure.face_mesh_provider import MediaPipeFaceMesh

        self._face_mesh = MediaPipeFaceMesh()
        self._detectors = DetectorFactory.create_all(self._settings, exclude={"yolo_eye"})
        self._lm_shm = SharedMemory(name=self._landmarks_shm_name, create=False)
        self._lm_array = np.ndarray(self._landmarks_shape, dtype=np.float32, buffer=self._lm_shm.buf)
        self._sig_shm = SharedMemory(name=self._signal_shm_name, create=False)
        self._sig_array = np.ndarray((SIGNAL_BUFFER_SIZE,), dtype=np.float32, buffer=self._sig_shm.buf)
        self._missing_frames = 0
        self._prev_time = time.perf_counter()
        self._fps = 0.0
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _compute_head_pose(self, landmarks: Any) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Compute head pose angles and vectors for rendering."""
        fw, fh = landmarks.frame_width, landmarks.frame_height
        if self._camera_matrix is None:
            focal_length = fw
            center = (fw / 2.0, fh / 2.0)
            self._camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
        image_points = landmarks.get_pixel_coords(landmarks.face_oval_indices).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, self._camera_matrix, self._dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            logger.error("solvepnp_failed")
            return 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3)

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        pitch = (180 - pitch) if pitch > 0 else (-180 - pitch)
        yaw = -yaw

        return pitch, yaw, roll, rvec.flatten(), tvec.flatten()

    def process_frame(self) -> None:
        if not self._new_frame_event.wait(timeout=0.1):
            return
        frame = self._frame_reader.read()
        now = time.perf_counter()
        dt = now - self._prev_time
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps = (0.7 * self._fps + 0.3 * inst_fps) if self._fps > 0.0 else inst_fps
        self._prev_time = now
        landmarks = self._face_mesh.get_landmarks(frame)
        if landmarks is not None:
            self._missing_frames = 0
            lm_data = landmarks.landmarks
            rows = min(lm_data.shape[0], self._landmarks_shape[0])
            self._lm_array[:rows, :] = lm_data[:rows, :]
            signals = []
            for detector in self._detectors:
                s = detector.detect(landmarks=landmarks, frame=frame)
                signals.append(s)
            idx = 0
            for s in signals:
                if idx + SIGNAL_SLOTS_PER_DETECTOR <= SIGNAL_BUFFER_SIZE:
                    self._sig_array[idx] = s.value
                    self._sig_array[idx + 1] = s.score
                    self._sig_array[idx + 2] = 1.0 if s.is_triggered else 0.0
                    idx += SIGNAL_SLOTS_PER_DETECTOR
            pitch, yaw, roll, rvec, tvec = self._compute_head_pose(landmarks)
            self._sig_array[SIG_FACE_DETECTED] = 1.0
            self._sig_array[SIG_FRAME_WIDTH] = float(landmarks.frame_width)
            self._sig_array[SIG_FRAME_HEIGHT] = float(landmarks.frame_height)
            self._sig_array[SIG_FPS] = self._fps
            self._sig_array[SIG_PITCH] = pitch
            self._sig_array[SIG_YAW] = yaw
            self._sig_array[SIG_ROLL] = roll
            self._sig_array[SIG_RVEC_0] = rvec[0]
            self._sig_array[SIG_RVEC_1] = rvec[1]
            self._sig_array[SIG_RVEC_2] = rvec[2]
            self._sig_array[SIG_TVEC_0] = tvec[0]
            self._sig_array[SIG_TVEC_1] = tvec[1]
            self._sig_array[SIG_TVEC_2] = tvec[2]
        else:
            self._missing_frames += 1
            self._sig_array[SIG_FACE_DETECTED] = 0.0
            self._sig_array[SIG_FPS] = self._fps
        self._landmarks_ready_event.set()

    def cleanup(self) -> None:
        self._face_mesh.release()
        self._lm_shm.close()
        self._sig_shm.close()
