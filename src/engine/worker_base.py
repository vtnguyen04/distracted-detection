from __future__ import annotations

import signal
from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class WorkerProcess(ABC):
    """
    Base class for pipeline workers. Each subclass runs in its own OS process,
    bypassing the Python GIL for true parallelism.
    Subclasses must implement:
        setup()         - one-time initialization (load models, open devices)
        process_frame() - main loop body, called once per new frame
        cleanup()       - graceful teardown
    """

    def __init__(self, running: Any, worker_name: str = "worker") -> None:
        self._running = running
        self._worker_name = worker_name

    @abstractmethod
    def setup(self) -> None:
        """Called once when the process starts. Initialize models, devices, etc."""

    @abstractmethod
    def process_frame(self) -> None:
        """Called in a loop. Read input SharedMemory, process, write output."""

    @abstractmethod
    def cleanup(self) -> None:
        """Called when the process is stopping. Release resources."""

    def run(self) -> None:
        """Main entry point — called by mp.Process(target=worker.run)."""
        import threading

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        import os

        logger.info(f"{self._worker_name}_started", pid=os.getpid(), thread_id=threading.get_native_id())
        try:
            self.setup()
            while self._running.value:
                self.process_frame()
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"{self._worker_name}_crashed", error=str(e))
        finally:
            self.cleanup()
            logger.info(f"{self._worker_name}_stopped")
