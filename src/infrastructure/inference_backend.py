from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
logger = structlog.get_logger()


class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str | Path, **kwargs: Any) -> None: ...
    @abstractmethod
    def predict(self, frame: NDArray[np.uint8], confidence: float = 0.4) -> list[dict[str, Any]]: ...
    @abstractmethod
    def release(self) -> None: ...
    @property
    @abstractmethod
    def backend_name(self) -> str: ...


from src.config.constants import YOLO_CLASS_NAMES


class OnnxBackend(InferenceBackend):
    def __init__(self) -> None:
        self._session: Any = None

    def load_model(self, model_path: str | Path, **kwargs: Any) -> None:
        try:
            import ctypes
            import os
            import site

            try:
                # Force-load nvidia pip binaries into global memory so onnxruntime can find them
                for sp in site.getsitepackages():
                    nv_dir = os.path.join(sp, "nvidia")
                    if os.path.isdir(nv_dir):
                        for lib_name in ["cudnn", "cublas"]:
                            lib_path = os.path.join(nv_dir, lib_name, "lib")
                            if os.path.isdir(lib_path):
                                for f in os.listdir(lib_path):
                                    if f.endswith(".so.9") or f.endswith(".so.12"):
                                        try:
                                            ctypes.CDLL(os.path.join(lib_path, f), mode=ctypes.RTLD_GLOBAL)
                                        except Exception:
                                            pass
            except Exception as _ignored:
                pass
            import onnxruntime as ort

        except ImportError as e:
            msg = "onnxruntime not installed. Install with: uv add onnxruntime"
            raise ImportError(msg) from e

        device = kwargs.get("device", "cuda:0").lower()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        logger.info("onnx_model_loaded", path=str(model_path), providers=self._session.get_providers())

    def predict(self, frame: NDArray[np.uint8], confidence: float = 0.4) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        h, w = (self._input_shape[2], self._input_shape[3])
        resized = cv2.resize(frame, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._parse_yolo_output(outputs[0], confidence, frame.shape[:2], (h, w))

    def release(self) -> None:
        self._session = None
        logger.info("onnx_backend_released")

    @property
    def backend_name(self) -> str:
        return "onnx"

    @staticmethod
    def _parse_yolo_output(
        output: NDArray[np.float32], conf_threshold: float, original_shape: tuple[int, int], net_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        import cv2

        detections: list[dict[str, Any]] = []
        predictions = output[0].T
        scale_x = original_shape[1] / net_shape[1]
        scale_y = original_shape[0] / net_shape[0]
        boxes = []
        scores = []
        class_ids = []
        for pred in predictions:
            x_c, y_c, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < conf_threshold:
                continue
            x1 = (x_c - w / 2) * scale_x
            y1 = (y_c - h / 2) * scale_y
            bw = w * scale_x
            bh = h * scale_y
            boxes.append([int(x1), int(y1), int(bw), int(bh)])
            scores.append(conf)
            class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
        for i in indices:
            idx = int(i) if np.isscalar(i) else int(i[0]) if isinstance(i, np.ndarray) and i.size > 0 else int(i)
            bx, by, bw, bh = boxes[idx]
            detections.append(
                {
                    "bbox": (bx, by, bx + bw, by + bh),
                    "confidence": scores[idx],
                    "class_id": class_ids[idx],
                    "class_name": YOLO_CLASS_NAMES.get(class_ids[idx], "Unknown"),
                }
            )
        return detections


class OpenVinoBackend(InferenceBackend):
    def __init__(self) -> None:
        self._compiled_model: Any = None
        self._input_layer: Any = None
        self._output_layer: Any = None

    def load_model(self, model_path: str | Path, **kwargs: Any) -> None:
        try:
            import openvino as ov
        except ImportError as e:
            msg = "openvino not installed. Install with: uv add openvino"
            raise ImportError(msg) from e
        device = str(kwargs.get("device", "AUTO")).upper()
        core = ov.Core()
        model = core.read_model(str(model_path))
        ov_config: dict[str, str] = {}
        available_devices = core.available_devices
        if "GPU" in device or ("AUTO" in device and "GPU" in available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES", "GPU_HOST_TASK_PRIORITY": "HIGH"}
        self._compiled_model = core.compile_model(model, device, ov_config)
        self._input_layer = self._compiled_model.input(0)
        self._output_layer = self._compiled_model.output(0)
        logger.info("openvino_model_loaded", path=str(model_path), device=device, available_devices=available_devices)

    def predict(self, frame: NDArray[np.uint8], confidence: float = 0.4) -> list[dict[str, Any]]:
        if self._compiled_model is None:
            return []
        input_shape = self._input_layer.shape
        h, w = (input_shape[2], input_shape[3])
        resized = cv2.resize(frame, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        result = self._compiled_model(blob)
        output = result[self._output_layer]
        return OnnxBackend._parse_yolo_output(output, confidence, frame.shape[:2], (h, w))

    def release(self) -> None:
        self._compiled_model = None
        logger.info("openvino_backend_released")

    @property
    def backend_name(self) -> str:
        return "openvino"


class UltralyticsBackend(InferenceBackend):
    def __init__(self) -> None:
        self._model: Any = None

    def load_model(self, model_path: str | Path, **kwargs: Any) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            msg = "ultralytics not installed. Install with: uv add ultralytics"
            raise ImportError(msg) from e

        self._model = YOLO(str(model_path), task="detect")
        device = kwargs.get("device", "cpu")
        self._model.to(device)
        self._device = device

        logger.info("ultralytics_model_loaded", path=str(model_path), device=device)

    def predict(self, frame: NDArray[np.uint8], confidence: float = 0.4) -> list[dict[str, Any]]:
        if self._model is None:
            return []
        results = self._model(frame, conf=confidence, verbose=False, max_det=10)
        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": YOLO_CLASS_NAMES.get(cls_id, "Unknown"),
                    }
                )
        return detections

    def release(self) -> None:
        self._model = None
        logger.info("ultralytics_backend_released")

    @property
    def backend_name(self) -> str:
        return "ultralytics"


class TensorrtBackend(InferenceBackend):
    """
    Custom high-performance TensorRT backend.
    Adapted from engine_utils.py / tensor_engine.py reference architecture:
    - Pre-allocated persistent host & device buffers (zero per-frame malloc)
    - Pinned (page-locked) host memory for DMA transfer
    - In-place numpy preprocessing (no intermediate copies)
    - Async CUDA stream execution via execute_async_v3
    - Persistent output buffer to avoid repeated GPU→CPU sync overhead
    """

    def __init__(self) -> None:
        import tensorrt as trt

        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._trt_logger, namespace="")
        self._runtime = trt.Runtime(self._trt_logger)
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._is_ready = False
        self._d_input: Any = None
        self._d_output: Any = None
        self._host_input: Any = None
        self._host_output: Any = None
        self._im_float: Any = None
        self._input_shape: tuple[int, ...] = ()
        self._output_shape: tuple[int, ...] = ()
        self._net_h = 0
        self._net_w = 0

    def load_model(self, model_path: str | Path, **kwargs: Any) -> None:
        import json
        import struct

        import tensorrt as trt
        import torch

        with open(model_path, "rb") as f:
            magic = f.read(4)
            magic_int = struct.unpack("<I", magic)[0]
            if magic_int < 10000:
                meta_bytes = f.read(magic_int)
                try:
                    json.loads(meta_bytes.decode("utf-8"))
                except Exception:
                    f.seek(0)
            else:
                f.seek(0)
            engine_bytes = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            msg = "Failed to deserialize TensorRT engine."
            raise RuntimeError(msg)
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        imgsz = kwargs.get("imgsz", (320, 320))
        self._net_h, self._net_w = imgsz[0], imgsz[1]
        self._input_shape = (1, 3, self._net_h, self._net_w)
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self._context.set_input_shape(name, self._input_shape)
                self._d_input = torch.empty(self._input_shape, dtype=torch.float32, device="cuda")
                self._context.set_tensor_address(name, self._d_input.data_ptr())
            else:
                self._output_shape = tuple(self._context.get_tensor_shape(name))
                self._d_output = torch.empty(self._output_shape, dtype=torch.float32, device="cuda")
                self._context.set_tensor_address(name, self._d_output.data_ptr())
        self._host_input = torch.empty(self._input_shape, dtype=torch.float32, device="cpu").pin_memory()
        self._host_output = torch.empty(self._output_shape, dtype=torch.float32, device="cpu").pin_memory()
        self._im_float = np.zeros((3, self._net_h, self._net_w), dtype=np.float32)
        self._is_ready = True
        self._loaded_path = str(model_path)
        logger.info(
            "tensorrt_custom_backend_initialized",
            path=str(model_path),
            input_shape=self._input_shape,
            output_shape=self._output_shape,
        )

    def predict(self, frame: NDArray[np.uint8], confidence: float = 0.4) -> list[dict[str, Any]]:
        if not self._is_ready:
            return []
        import torch

        h, w = self._net_h, self._net_w
        resized = cv2.resize(frame, (w, h))
        im_transposed = np.ascontiguousarray(resized.transpose(2, 0, 1)[::-1])
        np.multiply(im_transposed, 1.0 / 255.0, out=self._im_float)
        self._host_input[0].copy_(torch.from_numpy(self._im_float))
        self._d_input.copy_(self._host_input, non_blocking=True)
        self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._host_output.copy_(self._d_output, non_blocking=True)
        self._stream.synchronize()
        predictions = self._host_output.numpy()
        return OnnxBackend._parse_yolo_output(predictions, confidence, frame.shape[:2], (h, w))

    def release(self) -> None:
        self._context = None
        self._engine = None
        self._d_input = None
        self._d_output = None
        self._host_input = None
        self._host_output = None
        self._im_float = None
        self._is_ready = False
        logger.info("tensorrt_custom_backend_released")

    @property
    def backend_name(self) -> str:
        return "tensorrt"


def create_backend(backend_type: str, model_path: str | Path, **kwargs: Any) -> InferenceBackend:
    backends = {
        "onnx": OnnxBackend,
        "openvino": OpenVinoBackend,
        "ultralytics": UltralyticsBackend,
        "cuda": UltralyticsBackend,
        "tensorrt": TensorrtBackend,
    }
    cls = backends.get(backend_type.lower())
    if cls is None:
        msg = f"Unknown backend: {backend_type}. Available: {list(backends.keys())}"
        raise ValueError(msg)
    if backend_type.lower() in ["onnx", "openvino", "tensorrt"] and str(model_path).endswith(".pt"):
        from src.infrastructure.model_exporter import ModelExporter

        try:
            model_path = ModelExporter.auto_export(model_path, backend_type.lower())
        except Exception as e:
            logger.warning("auto_export_failed", error=str(e))

    if backend_type.lower() == "tensorrt" and str(model_path).endswith(".onnx"):
        engine_path = Path(str(model_path).replace(".onnx", ".engine"))
        if engine_path.exists():
            model_path = str(engine_path)

    backend = cls()
    backend.load_model(model_path, **kwargs)
    return backend
