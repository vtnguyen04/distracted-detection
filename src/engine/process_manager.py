from __future__ import annotations

import multiprocessing as mp
import signal
import threading
from typing import TYPE_CHECKING, Any

import structlog

from src.infrastructure.shared_memory_manager import SharedMemoryManager

if TYPE_CHECKING:
    from src.config.settings import AppSettings
logger = structlog.get_logger()


class ProcessManager:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._processes: list[Any] = []
        self._threads: list[threading.Thread] = []
        self._shm = SharedMemoryManager(frame_shape=(480, 640, 3))
        self._ctx = mp.get_context("fork")
        self._running = self._ctx.Value("i", 1)
        import os

        self._main_pid = os.getpid()

    def start(self) -> None:
        logger.info(
            "process_manager_starting",
            mode="multiprocessing_4workers"
            if self._settings.pipeline.use_multiprocessing
            else "threading_single_process",
        )
        ctx = self._ctx
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        from src.engine.capture_worker import CaptureWorker
        from src.engine.inference_worker import InferenceWorker
        from src.engine.mediapipe_worker import MediaPipeWorker
        from src.engine.orchestrator_worker import OrchestratorWorker
        from src.infrastructure.shared_memory_manager import SharedMemoryManager

        cap_worker = CaptureWorker(
            running=self._running,
            camera_settings=self._settings.camera,
            frame_writer=self._shm.frame_writer,
            new_frame_event=self._shm.state.get_event("new_frame"),
        )
        t_cap = threading.Thread(target=cap_worker.run, name="CaptureThread", daemon=True)
        self._threads.append(t_cap)
        import numpy as np

        from src.infrastructure.shared_memory_manager import SharedFrameReader

        mp_worker = MediaPipeWorker(
            running=self._running,
            settings=self._settings,
            frame_reader=SharedFrameReader(name=self._shm.frame_writer.name, shape=(480, 640, 3), dtype=np.uint8),
            new_frame_event=self._shm.state.get_event("new_frame"),
            landmarks_shm_name=self._shm.landmarks_shm_name,
            landmarks_shape=SharedMemoryManager.LANDMARKS_SHAPE,
            signal_shm_name=self._shm.signal_shm_name,
            landmarks_ready_event=self._shm.state.get_event("landmarks_ready"),
        )
        if self._settings.pipeline.use_multiprocessing:
            p_mp = ctx.Process(target=mp_worker.run, name="MediaPipeProcess")
            self._processes.append(p_mp)
        else:
            self._threads.append(threading.Thread(target=mp_worker.run, name="MediaPipeThread", daemon=True))

        inf_worker = InferenceWorker(
            running=self._running,
            settings=self._settings,
            frame_reader=SharedFrameReader(name=self._shm.frame_writer.name, shape=(480, 640, 3), dtype=np.uint8),
            new_frame_event=self._shm.state.get_event("new_frame"),
            detection_shm_name=self._shm.detection_shm_name,
            detections_ready_event=self._shm.state.get_event("detections_ready"),
            landmarks_shm_name=self._shm.landmarks_shm_name,
            landmarks_shape=SharedMemoryManager.LANDMARKS_SHAPE,
        )
        if self._settings.pipeline.use_multiprocessing:
            p_inf = ctx.Process(target=inf_worker.run, name="InferenceProcess")
            self._processes.append(p_inf)
        else:
            self._threads.append(threading.Thread(target=inf_worker.run, name="InferenceThread", daemon=True))

        orch_worker = OrchestratorWorker(
            running=self._running,
            settings=self._settings,
            frame_reader=SharedFrameReader(name=self._shm.frame_writer.name, shape=(480, 640, 3), dtype=np.uint8),
            result_writer=self._shm.result_writer,
            landmarks_shm_name=self._shm.landmarks_shm_name,
            landmarks_shape=SharedMemoryManager.LANDMARKS_SHAPE,
            signal_shm_name=self._shm.signal_shm_name,
            detection_shm_name=self._shm.detection_shm_name,
            landmarks_ready_event=self._shm.state.get_event("landmarks_ready"),
            detections_ready_event=self._shm.state.get_event("detections_ready"),
            new_frame_event=self._shm.state.get_event("new_frame"),
            show_frame_event=self._shm.state.get_event("show_frame"),
            state=self._shm.state,
        )
        if self._settings.pipeline.use_multiprocessing:
            p_orch = ctx.Process(target=orch_worker.run, name="OrchestratorProcess")
            self._processes.append(p_orch)
        else:
            self._threads.append(threading.Thread(target=orch_worker.run, name="OrchestratorThread", daemon=True))
        for p in self._processes:
            p.start()
        for t in self._threads:
            t.start()

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        logger.info("Received termination signal", signum=signum)
        self.stop()

    def stop(self) -> None:
        import os

        if os.getpid() != getattr(self, "_main_pid", 0):
            return
        logger.info("process_manager_stopping")
        self._running.value = 0
        os.system("pkill aplay; pkill paplay")
        for p in self._processes:
            if p.is_alive():
                p.join(timeout=2.0)
                if p.is_alive():
                    if p._popen is not None:
                        p.terminate()
                    p.join(timeout=1.0)
                    if p.is_alive():
                        if p._popen is not None:
                            p.kill()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._processes.clear()
        self._threads.clear()
        self._shm.cleanup()
        logger.info("process_manager_stopped")
