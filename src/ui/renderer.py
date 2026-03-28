from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.config.constants import (
    COLOR_AXIS_X,
    COLOR_AXIS_Y,
    COLOR_AXIS_Z,
    COLOR_DARK_BG,
    COLOR_GREEN,
    COLOR_MESH_FACE,
    COLOR_RED,
    COLOR_RETICLE,
    COLOR_WHITE,
    COLOR_YELLOW,
    HEAD_POSE_MODEL_POINTS_3D,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.models import DetectionSignal, FaceLandmarks


class FaceMeshRenderer:
    """Draws full face mesh tessellation grid with highlighted contours."""

    def render(self, frame: NDArray[np.uint8], landmarks: FaceLandmarks) -> None:
        """Draw full mesh tessellation grid, then highlighted contours on top."""
        from src.config.mesh_dump import FACEMESH_TESSELATION

        lm = landmarks.landmarks
        landmarks_px = landmarks.get_pixel_coords(list(range(len(lm))))
        mesh_overlay = frame.copy()
        for edge in FACEMESH_TESSELATION:
            p1 = (int(landmarks_px[edge[0]][0]), int(landmarks_px[edge[0]][1]))
            p2 = (int(landmarks_px[edge[1]][0]), int(landmarks_px[edge[1]][1]))
            cv2.line(mesh_overlay, p1, p2, COLOR_MESH_FACE, 1, cv2.LINE_AA)
        cv2.addWeighted(mesh_overlay, 0.4, frame, 0.6, 0, frame)
        for idx_list in [landmarks.left_eye_indices, landmarks.right_eye_indices, landmarks.mouth_indices]:
            pts = landmarks.get_pixel_coords(idx_list)
            for pt in pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, COLOR_MESH_FACE, -1)


class HeadPoseRenderer:
    """Draws sleek glowing orange target reticle and 3D head pose axes."""

    _MODEL_POINTS = np.array(HEAD_POSE_MODEL_POINTS_3D, dtype=np.float64)
    _AXIS_LENGTH = 80.0

    def render(
        self,
        frame: NDArray[np.uint8],
        landmarks: FaceLandmarks,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        rotation_vec: NDArray[np.float64] | None = None,
        translation_vec: NDArray[np.float64] | None = None,
    ) -> None:
        """Draw sci-fi reticle and subtle axes."""
        fw, fh = landmarks.frame_width, landmarks.frame_height
        focal_length = fw
        center = (fw / 2.0, fh / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        if rotation_vec is not None and translation_vec is not None:
            rvec = rotation_vec
        else:
            image_points = landmarks.get_pixel_coords(landmarks.face_oval_indices).astype(np.float64)
            success, rvec, _tvec = cv2.solvePnP(
                self._MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return
        nose_tip = landmarks.get_pixel_coords([landmarks.nose_tip_index])[0]
        face_oval = landmarks.get_pixel_coords(landmarks.face_oval_indices)
        min_x, max_x = np.min(face_oval[:, 0]), np.max(face_oval[:, 0])
        min_y, max_y = np.min(face_oval[:, 1]), np.max(face_oval[:, 1])
        radius = int(max((max_x - min_x), (max_y - min_y)) * 0.9)
        origin = (int(nose_tip[0]), int(nose_tip[1] - radius * 0.35))
        cv2.circle(frame, origin, radius, COLOR_RETICLE, 2, cv2.LINE_AA)
        cv2.circle(frame, origin, int(radius * 1.15), COLOR_RETICLE, 1, cv2.LINE_AA)
        tick_len = 15
        cv2.line(
            frame,
            (origin[0], origin[1] - radius),
            (origin[0], origin[1] - radius + tick_len),
            COLOR_RETICLE,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (origin[0], origin[1] + radius),
            (origin[0], origin[1] + radius - tick_len),
            COLOR_RETICLE,
            2,
            cv2.LINE_AA,
        )
        if rvec is None or np.all(rvec == 0):
            return

        rvec_adj = np.array(rvec, dtype=np.float64).copy().ravel()
        rvec_adj[1] = -rvec_adj[1]  # The reference repo negates the Y vector

        rotation_matrix, _ = cv2.Rodrigues(rvec_adj)

        axes_points = np.array([[-1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, -2.0, 0]], dtype=np.float64)

        axes_points = rotation_matrix @ axes_points
        axes_points = (axes_points[:2, :] * 50).astype(int)

        nose_o = (int(nose_tip[0]), int(nose_tip[1]))
        end_x = (int(nose_o[0] + axes_points[0, 0]), int(nose_o[1] + axes_points[1, 0]))
        end_y = (int(nose_o[0] + axes_points[0, 1]), int(nose_o[1] + axes_points[1, 1]))
        end_z = (int(nose_o[0] + axes_points[0, 2]), int(nose_o[1] + axes_points[1, 2]))

        cv2.line(frame, nose_o, end_x, COLOR_AXIS_X, 2, cv2.LINE_AA)
        cv2.line(frame, nose_o, end_y, COLOR_AXIS_Y, 2, cv2.LINE_AA)
        cv2.line(frame, nose_o, end_z, COLOR_AXIS_Z, 2, cv2.LINE_AA)


class HudRenderer:
    """Renders textual heads-up display and signal gauges (Now Disabled - Handled by PySide)."""

    def render(
        self,
        frame: NDArray[np.uint8],
        signals: list[DetectionSignal],
        distraction_score: float,
        driver_state_name: str,
        backend_name: str,
        fps: float,
    ) -> None:
        pass


class YoloBboxRenderer:
    """Draws sleek, modern YOLO detection bounding boxes with minimalistic labels."""

    def render(self, frame: NDArray[np.uint8], signals: list[DetectionSignal]) -> None:
        """Draw minimalist bounding boxes for detections."""
        for signal in signals:
            if signal.detector_name != "yolo_eye" or "detections" not in signal.metadata:
                continue
            face_bbox = signal.metadata.get("face_bbox")
            if not face_bbox:
                continue
            fx1, fy1, _fx2, _fy2 = face_bbox
            for det in signal.metadata["detections"]:
                if det["class_name"] == "Face":
                    continue
                x1, y1, x2, y2 = det["bbox"]
                x1, y1 = x1 + fx1, y1 + fy1
                x2, y2 = x2 + fx1, y2 + fy1
                cls_name = det["class_name"]
                if cls_name == "Eye closed":
                    color = COLOR_RED
                elif cls_name == "Mouth":
                    color = COLOR_WHITE
                elif cls_name == "Face":
                    color = COLOR_MESH_FACE
                else:
                    color = (255, 0, 200)
                length = 8
                thick = 2
                cv2.line(frame, (x1, y1), (x1 + length, y1), color, thick)
                cv2.line(frame, (x1, y1), (x1, y1 + length), color, thick)
                cv2.line(frame, (x2, y1), (x2 - length, y1), color, thick)
                cv2.line(frame, (x2, y1), (x2, y1 + length), color, thick)
                cv2.line(frame, (x1, y2), (x1 + length, y2), color, thick)
                cv2.line(frame, (x1, y2), (x1, y2 - length), color, thick)
                cv2.line(frame, (x2, y2), (x2 - length, y2), color, thick)
                cv2.line(frame, (x2, y2), (x2, y2 - length), color, thick)
                label = f"{cls_name.split()[0]} {det['confidence']:.2f}"
                cv2.putText(
                    frame, label, (x1, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA
                )


class GaugeRenderer:
    """Draws the distraction gauge bar at the bottom."""

    def render(self, frame: NDArray[np.uint8], score: float, w: int) -> None:
        """Draw distraction progress bar."""
        bar_h = 20
        bar_y = frame.shape[0] - bar_h - 10
        bar_w = w - 20
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), COLOR_DARK_BG, -1)
        fill_w = int(bar_w * min(score, 1.0))
        color = COLOR_GREEN if score < 0.5 else COLOR_YELLOW if score < 0.7 else COLOR_RED
        cv2.rectangle(frame, (10, bar_y), (10 + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), COLOR_WHITE, 1)
        cv2.putText(
            frame, f"Distraction: {score:.0%}", (15, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1
        )


class StateBannerRenderer:
    """Draws the driver state banner."""

    _COLORS = {"ALERT": COLOR_GREEN, "WARNING": COLOR_YELLOW, "DISTRACTED": COLOR_RED}

    def render(self, frame: NDArray[np.uint8], state: str, w: int) -> None:
        """Draw state text in top-right corner."""
        color = self._COLORS.get(state, COLOR_WHITE)
        cv2.putText(frame, state, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


class Renderer:
    """Composite renderer using Strategy pattern — delegates to specialized render layers."""

    def __init__(self) -> None:
        self._face_mesh = FaceMeshRenderer()
        self._head_pose = HeadPoseRenderer()
        self._yolo_bbox = YoloBboxRenderer()

    def render(
        self,
        frame: NDArray[np.uint8],
        landmarks: FaceLandmarks | None,
        signals: list[DetectionSignal],
        distraction_score: float,
        driver_state_name: str,
        fps: float,
        backend_name: str = "cpu",
        head_pose_data: dict[str, float] | None = None,
    ) -> NDArray[np.uint8]:
        """Compose all render layers into the final annotated frame."""
        annotated = frame.copy()
        if landmarks is not None:
            self._face_mesh.render(annotated, landmarks)
            hp = head_pose_data or {}
            self._head_pose.render(
                annotated,
                landmarks,
                pitch=hp.get("pitch", 0.0),
                yaw=hp.get("yaw", 0.0),
                roll=hp.get("roll", 0.0),
                rotation_vec=hp.get("rvec"),
                translation_vec=hp.get("tvec"),
            )
        self._yolo_bbox.render(annotated, signals)
        return annotated
