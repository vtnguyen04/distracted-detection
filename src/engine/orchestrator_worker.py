from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from src.config.constants import (
    DET_CONFIDENCE,
    DET_CROP_X1,
    DET_CROP_X2,
    DET_CROP_Y1,
    DET_CROP_Y2,
    DET_SCORE,
    DET_TRIGGERED,
    DET_VALUE,
    DETECTION_BUFFER_SIZE,
    DETECTION_HEADER_SIZE,
    DETECTION_STRIDE,
    ID_TO_CLASS_NAME,
    LANDMARK_DETECTOR_NAMES,
    MAX_DETECTIONS,
    SIG_FACE_DETECTED,
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


class OrchestratorWorker(WorkerProcess):
    """Fuses signals from MediaPipe and Inference workers, renders the annotated frame."""

    def __init__(
        self,
        running: Any,
        settings: Any,
        frame_reader: Any,
        result_writer: Any,
        landmarks_shm_name: str,
        landmarks_shape: tuple[int, ...],
        signal_shm_name: str,
        detection_shm_name: str,
        landmarks_ready_event: Any,
        detections_ready_event: Any,
        new_frame_event: Any,
        show_frame_event: Any,
        state: Any,
    ) -> None:
        super().__init__(running, worker_name="orchestrator_worker")
        self._settings = settings
        self._frame_reader = frame_reader
        self._result_writer = result_writer
        self._landmarks_shm_name = landmarks_shm_name
        self._landmarks_shape = landmarks_shape
        self._signal_shm_name = signal_shm_name
        self._detection_shm_name = detection_shm_name
        self._landmarks_ready_event = landmarks_ready_event
        self._detections_ready_event = detections_ready_event
        self._new_frame_event = new_frame_event
        self._show_frame_event = show_frame_event
        self._state = state

    def setup(self) -> None:
        from multiprocessing.shared_memory import SharedMemory

        from src.detectors.signal_fusion import SignalFusion
        from src.engine.state_machine import DriverStateMachine
        from src.infrastructure.alert_manager import AlertManager
        from src.ui.renderer import Renderer

        self._fusion = SignalFusion()
        self._state_machine = DriverStateMachine(self._settings.state_machine)
        self._alert = AlertManager(self._settings.alert)
        self._renderer = Renderer()
        self._lm_shm = SharedMemory(name=self._landmarks_shm_name, create=False)
        self._lm_array = np.ndarray(self._landmarks_shape, dtype=np.float32, buffer=self._lm_shm.buf)
        self._sig_shm = SharedMemory(name=self._signal_shm_name, create=False)
        self._sig_array = np.ndarray((SIGNAL_BUFFER_SIZE,), dtype=np.float32, buffer=self._sig_shm.buf)
        self._det_shm = SharedMemory(name=self._detection_shm_name, create=False)
        self._det_array = np.ndarray((DETECTION_BUFFER_SIZE,), dtype=np.float32, buffer=self._det_shm.buf)
        self._missing_frames = 0
        self._frame_count = 0
        self._prof_wait = 0.0
        self._prof_fuse = 0.0
        self._prof_render = 0.0
        self._prev_time = time.perf_counter()
        self._display_fps = 0.0
        w = self._settings.weights
        self._all_detector_weights = {
            "ear": w.ear,
            "mar": w.mar,
            "head_pose": w.head_pose,
            "perclos": w.perclos,
            "blink_rate": w.blink_rate,
            "yolo_eye": w.yolo_eye,
        }

    def process_frame(self) -> None:
        from src.domain.models import DetectionSignal, FaceLandmarks

        _t0 = time.perf_counter()
        lm_ok = self._landmarks_ready_event.wait(timeout=0.2)
        det_ok = self._detections_ready_event.wait(timeout=0.2)

        if not lm_ok and not det_ok:
            return

        self._landmarks_ready_event.clear()
        self._detections_ready_event.clear()
        _t_wait = time.perf_counter()
        frame = self._frame_reader.read()
        now = time.perf_counter()
        dt = now - self._prev_time
        if dt > 0:
            inst_fps = 1.0 / dt
            self._display_fps = (0.8 * self._display_fps + 0.2 * inst_fps) if self._display_fps > 0 else inst_fps
        self._prev_time = now
        fps = self._display_fps
        face_detected = self._sig_array[SIG_FACE_DETECTED] > 0.5
        self._state.set("fps", int(fps))
        self._state.set("face_detected", 1 if face_detected else 0)
        signals: list[DetectionSignal] = []
        landmarks = None
        head_pose_data: dict[str, float] | None = None
        if face_detected:
            self._missing_frames = 0
            fw = int(self._sig_array[SIG_FRAME_WIDTH])
            fh = int(self._sig_array[SIG_FRAME_HEIGHT])
            landmarks = FaceLandmarks(
                landmarks=np.array(self._lm_array),
                frame_width=fw,
                frame_height=fh,
            )
            idx = 0
            for det_name in LANDMARK_DETECTOR_NAMES:
                if idx + SIGNAL_SLOTS_PER_DETECTOR <= SIGNAL_BUFFER_SIZE:
                    signals.append(
                        DetectionSignal(
                            detector_name=det_name,
                            value=float(self._sig_array[idx]),
                            score=float(self._sig_array[idx + 1]),
                            is_triggered=self._sig_array[idx + 2] > 0.5,
                        )
                    )
                    idx += SIGNAL_SLOTS_PER_DETECTOR
            head_pose_data = {
                "pitch": float(self._sig_array[SIG_PITCH]),
                "yaw": float(self._sig_array[SIG_YAW]),
                "roll": float(self._sig_array[SIG_ROLL]),
                "rvec": np.array(
                    [self._sig_array[SIG_RVEC_0], self._sig_array[SIG_RVEC_1], self._sig_array[SIG_RVEC_2]],
                    dtype=np.float64,
                ),
                "tvec": np.array(
                    [self._sig_array[SIG_TVEC_0], self._sig_array[SIG_TVEC_1], self._sig_array[SIG_TVEC_2]],
                    dtype=np.float64,
                ),
            }
            yolo_value = float(self._det_array[DET_VALUE])
            yolo_score = float(self._det_array[DET_SCORE])
            yolo_triggered = self._det_array[DET_TRIGGERED] > 0.5
            yolo_confidence = float(self._det_array[DET_CONFIDENCE])
            fx1 = int(self._det_array[DET_CROP_X1])
            fy1 = int(self._det_array[DET_CROP_Y1])
            fx2 = int(self._det_array[DET_CROP_X2])
            fy2 = int(self._det_array[DET_CROP_Y2])
            face_bbox = (fx1, fy1, fx2, fy2)
            det_list = []
            for i in range(MAX_DETECTIONS):
                offset = DETECTION_HEADER_SIZE + i * DETECTION_STRIDE
                conf = float(self._det_array[offset + 5])
                if conf <= 0.0:
                    break
                det_list.append(
                    {
                        "bbox": (
                            int(self._det_array[offset]),
                            int(self._det_array[offset + 1]),
                            int(self._det_array[offset + 2]),
                            int(self._det_array[offset + 3]),
                        ),
                        "class_name": ID_TO_CLASS_NAME.get(int(self._det_array[offset + 4]), "Unknown"),
                        "confidence": conf,
                    }
                )
            signals.append(
                DetectionSignal(
                    detector_name="yolo_eye",
                    value=yolo_value,
                    score=yolo_score,
                    is_triggered=yolo_triggered,
                    confidence=yolo_confidence,
                    metadata={"detections": det_list, "face_bbox": face_bbox},
                )
            )
            distracted_score = self._fusion.fuse(signals, self._all_detector_weights)
        else:
            self._missing_frames += 1
            fps_val = fps if fps > 0 else 30.0
            distracted_score = 1.0 if self._missing_frames > (1.5 * fps_val) else 0.0
        _t_fuse = time.perf_counter()
        self._state.set("is_distracted", int(distracted_score))
        self._state.set("distraction_score", float(distracted_score))
        self._state_machine.update(distracted_score)
        state_name = self._state_machine.state.name
        annotated = self._renderer.render(
            frame=frame,
            landmarks=landmarks,
            signals=signals,
            distraction_score=distracted_score,
            driver_state_name=state_name,
            fps=fps,
            backend_name=self._settings.inference.backend,
            head_pose_data=head_pose_data,
        )
        _t_render = time.perf_counter()
        self._result_writer.write(annotated)
        self._state.set("alert_level", int(self._state_machine.alert_level.value))
        if self._state_machine.alert_level.value >= 2:
            self._alert.play_alert(self._state_machine.alert_level)
        else:
            self._alert.stop_alert()
        self._show_frame_event.set()
        self._prof_wait += _t_wait - _t0
        self._prof_fuse += _t_fuse - _t_wait
        self._prof_render += _t_render - _t_fuse
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            n = 30
            logger.info(
                "pipeline_profile",
                wait_ms=round(self._prof_wait / n * 1000, 1),
                fuse_ms=round(self._prof_fuse / n * 1000, 1),
                render_ms=round(self._prof_render / n * 1000, 1),
                total_ms=round((self._prof_wait + self._prof_fuse + self._prof_render) / n * 1000, 1),
                fps=round(fps, 1),
            )
            self._prof_wait = self._prof_fuse = self._prof_render = 0.0

    def cleanup(self) -> None:
        self._alert.stop_alert()
        self._lm_shm.close()
        self._sig_shm.close()
        self._det_shm.close()
