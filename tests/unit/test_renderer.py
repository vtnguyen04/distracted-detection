import numpy as np
from src.domain.models import FaceLandmarks
from src.ui.renderer import Renderer


def test_renderer_initialization():
    renderer = Renderer()
    assert renderer is not None


def test_renderer_draw_frame():
    renderer = Renderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create fake landmarks representing a centered face
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[:, 0] = 0.5  # x
    lm[:, 1] = 0.5  # y
    lm[:, 2] = 0.1  # z

    landmarks = FaceLandmarks(landmarks=lm, frame_width=640, frame_height=480)

    annotated = renderer.render(
        frame=frame,
        landmarks=landmarks,
        signals=[],
        distraction_score=0.9,
        driver_state_name="SAFE",
        fps=30.0,
        backend_name="cpu",
        head_pose_data={"pitch": 10.0, "yaw": -5.0, "roll": 0.0, "rvec": np.zeros(3), "tvec": np.zeros(3)},
    )

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    assert np.any(annotated > 0), "Renderer must output modified pixels into the buffer"
