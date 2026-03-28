from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from src.detectors.signal_fusion import SignalFusion
from src.domain.enums import AlertLevel
from src.domain.models import DetectionResult
from src.utils.profiler import InferenceProfiler

if TYPE_CHECKING:
    from src.config.settings import AppSettings
    from src.domain.protocols import (
        AlertProvider,
        CameraProvider,
        Detector,
        FaceMeshProvider,
        FrameRenderer,
    )
    from src.engine.state_machine import DriverStateMachine
logger = structlog.get_logger()


class DetectionPipeline:
    """Main detection pipeline — single-threaded for simplicity.
    For production multiprocessing, wrap this in a Process via ProcessManager.
    """

    def __init__(
        self,
        camera: CameraProvider,
        face_mesh: FaceMeshProvider,
        detectors: list[Detector],
        alert: AlertProvider,
        renderer: FrameRenderer,
        state_machine: DriverStateMachine,
        settings: AppSettings,
    ) -> None:
        self._camera = camera
        self._face_mesh = face_mesh
        self._detectors = detectors
        self._alert = alert
        self._renderer = renderer
        self._state_machine = state_machine
        self._settings = settings
        self._fusion = SignalFusion()
        self._running = False
        self._frame_count = 0
        self._profiler = InferenceProfiler(enabled=settings.pipeline.profiling)

    def run(self) -> None:
        """Start the detection loop. Blocks until stopped."""
        self._running = True
        prev_time = time.monotonic()
        fps = 0.0
        logger.info(
            "pipeline_started",
            detectors=[d.name for d in self._detectors],
            backend=self._settings.pipeline.model_backend,
        )
        try:
            while self._running and self._camera.is_opened:
                success, frame = self._camera.read_frame()
                if not success:
                    logger.warning("camera_frame_failed")
                    break
                self._frame_count += 1
                skip = self._settings.pipeline.frame_skip
                if skip > 0 and self._frame_count % (skip + 1) != 0:
                    continue
                now = time.monotonic()
                dt = now - prev_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps = (0.9 * fps + 0.1 * inst_fps) if fps > 0.0 else inst_fps
                prev_time = now
                self._profiler.tick_frame()
                t_mesh = time.perf_counter()
                landmarks = self._face_mesh.get_landmarks(frame)
                self._profiler.record("face_mesh", t_mesh)
                result = DetectionResult(fps=fps, face_detected=landmarks is not None)
                signals_collected = self._run_detectors(frame, landmarks, result)
                if signals_collected:
                    self._process_active_face(result)
                else:
                    self._handle_missing_face(result, fps)
                self._profiler.log_summary(interval_frames=100)
                self._render_ui(frame, landmarks, result, fps)
        except KeyboardInterrupt:
            logger.info("pipeline_interrupted")
        finally:
            self._cleanup()

    def _run_detectors(self, frame: Any, landmarks: Any, result: DetectionResult) -> bool:
        t_detect = time.perf_counter()
        collected = False
        for detector in self._detectors:
            if landmarks is None and not hasattr(detector, "detect_from_frame"):
                continue
            if hasattr(detector, "detect_from_frame"):
                signal = detector.detect_from_frame(frame, landmarks)
            else:
                signal = detector.detect(landmarks=landmarks, frame=frame)
            result.signals.append(signal)
            collected = True
        self._profiler.record("detectors", t_detect)
        return collected

    def _process_active_face(self, result: DetectionResult) -> None:
        """Process signals when a driver face is detected."""
        self._missing_face_frames = 0
        t_fuse = time.perf_counter()
        weight_dict = {d.name: d.weight for d in self._detectors}
        result.distraction_score = self._fusion.fuse(result.signals, weight_dict)
        state = self._state_machine.update(result.distraction_score)
        result.driver_state = state
        result.alert_level = self._state_machine.alert_level
        self._handle_alert(result.alert_level)
        self._profiler.record("fusion_state", t_fuse)

    def _handle_missing_face(self, result: DetectionResult, fps: float) -> None:
        """Handle scenarios where no face is detected."""
        self._missing_face_frames = getattr(self, "_missing_face_frames", 0) + 1
        fps_val = fps if fps > 0 else 30.0
        if self._missing_face_frames > (1.5 * fps_val):
            result.distraction_score = 1.0
            state = self._state_machine.update(1.0)
            result.driver_state = state
            result.alert_level = self._state_machine.alert_level
            self._handle_alert(result.alert_level)
        else:
            state = self._state_machine.update(0.0)
            self._alert.stop_alert()

    def _render_ui(self, frame: Any, landmarks: Any, result: DetectionResult, fps: float) -> None:
        import cv2

        if not self._settings.pipeline.show_ui:
            return
        annotated = self._renderer.render(
            frame=frame,
            landmarks=landmarks,
            signals=result.signals,
            distraction_score=result.distraction_score,
            driver_state_name=result.driver_state.name if result.driver_state else "N/A",
            fps=fps,
            backend_name=self._settings.inference.backend,
        )
        cv2.imshow("Distracted Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self._running = False

    def _handle_alert(self, level: AlertLevel) -> None:
        """Manage alert sound based on alert level."""
        if level in (AlertLevel.MEDIUM, AlertLevel.HIGH):
            self._alert.play_alert(level)
        elif level == AlertLevel.NONE:
            self._alert.stop_alert()

    def _cleanup(self) -> None:
        import cv2

        self._camera.release()
        self._face_mesh.release()
        self._alert.stop_alert()
        cv2.destroyAllWindows()
        logger.info("pipeline_stopped", total_frames=self._frame_count)
