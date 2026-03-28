from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np
import structlog
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.domain.models import FaceLandmarks

if TYPE_CHECKING:
    from numpy.typing import NDArray
logger = structlog.get_logger()
_DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


class MediaPipeFaceMesh:
    def __init__(
        self,
        model_path: str | Path | None = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            msg = f"FaceLandmarker model not found at: {resolved_path}. Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            raise FileNotFoundError(msg)
        base_options = mp_python.BaseOptions(model_asset_path=str(resolved_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_count = 0
        logger.info("face_landmarker_initialized", model=str(resolved_path))

    def get_landmarks(self, frame: NDArray[np.uint8]) -> FaceLandmarks | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._frame_count += 1
        timestamp_ms = int(self._frame_count * (1000 / 30))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32)
        h, w = frame.shape[:2]
        return FaceLandmarks(landmarks=landmarks_array, frame_width=w, frame_height=h)

    def release(self) -> None:
        if self._landmarker:
            self._landmarker.close()
            logger.info("face_landmarker_released")
