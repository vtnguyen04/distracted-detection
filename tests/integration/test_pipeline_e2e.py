import time

import numpy as np
import pytest
from src.config.settings import AppSettings
from src.engine.process_manager import ProcessManager


class FakeCamera:
    """A multiprocessing-safe dummy camera to simulate webcam input without hardware."""

    def __init__(self, *args, **kwargs):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def read_frame(self):
        # Simulately yield a 480p RGB frame
        time.sleep(0.01)  # Max 100fps simulation
        return True, self.frame

    def release(self):
        pass


@pytest.fixture
def e2e_settings():
    settings = AppSettings()
    settings.pipeline.use_multiprocessing = True
    settings.inference.backend = "onnx"
    settings.inference.device = "cpu"
    return settings


def test_pipeline_spawns_and_computes_fps(e2e_settings, mocker):
    mocker.patch("src.engine.capture_worker.Camera", new=FakeCamera)

    manager = ProcessManager(e2e_settings)
    manager.start()

    # Run the multithreaded architecture for enough time to pass frames
    time.sleep(3.5)

    fps = manager._shm.state.get("fps")

    manager.stop()

    assert fps is not None, "Pipeline FPS should be populated in SharedState"
    assert fps > 0.0, "Pipeline should be actively processing frames and computing FPS"
