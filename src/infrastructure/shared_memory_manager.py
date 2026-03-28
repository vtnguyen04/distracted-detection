from __future__ import annotations

import atexit
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray
logger = structlog.get_logger()


class SharedFrameWriter:
    def __init__(self, shape: tuple[int, ...], dtype: type = np.uint8) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._size = int(np.prod(shape)) * self._dtype.itemsize
        self._shm = SharedMemory(create=True, size=self._size)
        self._array = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        logger.info("shared_frame_writer_created", shape=shape, size_bytes=self._size)

    def write(self, data: NDArray[Any]) -> None:
        if data.shape != self._shape:
            import cv2

            h, w = self._shape[:2]
            data = cv2.resize(data, (w, h))
        np.copyto(self._array, data)

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
            logger.info("shared_frame_writer_released")
        except FileNotFoundError:
            pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def name(self) -> str:
        return self._shm.name


class SharedFrameReader:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: type = np.uint8) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._shm = SharedMemory(name=name, create=False)
        self._array = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        logger.info("shared_frame_reader_attached", shape=shape)

    def read(self) -> NDArray[Any]:
        return np.array(self._array)

    def close(self) -> None:
        self._shm.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape


class SharedState:
    def __init__(self) -> None:
        import multiprocessing as mp

        self._ctx = mp.get_context("fork")
        self._values: dict[str, Any] = {}
        self._events: dict[str, Any] = {}

    def add_value(self, name: str, typecode: str = "f", initial: float = 0.0) -> None:
        self._values[name] = self._ctx.Value(typecode, initial)

    def add_event(self, name: str) -> None:
        self._events[name] = self._ctx.Event()

    def get(self, name: str) -> float:
        v = self._values.get(name)
        if v is None:
            msg = f"Unknown shared value: {name}"
            raise KeyError(msg)
        with v.get_lock():
            return v.value

    def set(self, name: str, value: float) -> None:
        v = self._values.get(name)
        if v is None:
            msg = f"Unknown shared value: {name}"
            raise KeyError(msg)
        with v.get_lock():
            v.value = value

    def increment(self, name: str, delta: float = 1.0) -> None:
        v = self._values[name]
        with v.get_lock():
            v.value += delta

    def get_event(self, name: str) -> Any:
        return self._events[name]

    def get_value_raw(self, name: str) -> Any:
        return self._values[name]


class SharedMemoryManager:
    from src.config.constants import DETECTION_BUFFER_SIZE as DETECTION_BUFFER_SIZE
    from src.config.constants import LANDMARKS_SHAPE as LANDMARKS_SHAPE
    from src.config.constants import SIGNAL_BUFFER_SIZE as SIGNAL_BUFFER_SIZE

    def __init__(self, frame_shape: tuple[int, ...] = (320, 320, 3)) -> None:
        self.frame_writer = SharedFrameWriter(frame_shape, dtype=np.uint8)
        self.frame_reader = SharedFrameReader(name=self.frame_writer.name, shape=frame_shape, dtype=np.uint8)
        self.result_writer = SharedFrameWriter(frame_shape, dtype=np.uint8)
        self.result_reader = SharedFrameReader(name=self.result_writer.name, shape=frame_shape, dtype=np.uint8)
        lm_size = int(np.prod(self.LANDMARKS_SHAPE)) * np.dtype(np.float32).itemsize
        from multiprocessing.shared_memory import SharedMemory as SHM

        self._lm_shm = SHM(create=True, size=lm_size)
        self._lm_array = np.ndarray(self.LANDMARKS_SHAPE, dtype=np.float32, buffer=self._lm_shm.buf)
        self._lm_array[:] = 0.0
        sig_size = self.SIGNAL_BUFFER_SIZE * np.dtype(np.float32).itemsize
        self._sig_shm = SHM(create=True, size=sig_size)
        self._sig_array = np.ndarray((self.SIGNAL_BUFFER_SIZE,), dtype=np.float32, buffer=self._sig_shm.buf)
        self._sig_array[:] = 0.0
        det_size = self.DETECTION_BUFFER_SIZE * np.dtype(np.float32).itemsize
        self._det_shm = SHM(create=True, size=det_size)
        self._det_array = np.ndarray((self.DETECTION_BUFFER_SIZE,), dtype=np.float32, buffer=self._det_shm.buf)
        self._det_array[:] = 0.0
        self.state = SharedState()
        self.state.add_value("running", "i", 0)
        self.state.add_value("fps", "i", 0)
        self.state.add_value("frame_count", "i", 0)
        self.state.add_value("distraction_score", "f", 0.0)
        self.state.add_value("eye_closed_count", "f", 0.0)
        self.state.add_value("eye_open_count", "f", 0.0)
        self.state.add_value("is_distracted", "i", 0)
        self.state.add_value("eye_state", "f", 0.0)
        self.state.add_value("face_detected", "i", 0)
        self.state.add_value("alert_level", "i", 0)
        self.state.add_event("new_frame")
        self.state.add_event("show_frame")
        self.state.add_event("landmarks_ready")
        self.state.add_event("detections_ready")
        atexit.register(self.cleanup)
        logger.info("shared_memory_manager_initialized", frame_shape=frame_shape)

    @property
    def landmarks_shm_name(self) -> str:
        return self._lm_shm.name

    @property
    def signal_shm_name(self) -> str:
        return self._sig_shm.name

    @property
    def detection_shm_name(self) -> str:
        return self._det_shm.name

    def cleanup(self) -> None:
        self.frame_reader.close()
        self.frame_writer.close()
        self.result_reader.close()
        self.result_writer.close()
        try:
            self._lm_shm.close()
            self._lm_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            self._sig_shm.close()
            self._sig_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            self._det_shm.close()
            self._det_shm.unlink()
        except FileNotFoundError:
            pass
        logger.info("shared_memory_manager_cleaned_up")

    def __enter__(self) -> SharedMemoryManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()
