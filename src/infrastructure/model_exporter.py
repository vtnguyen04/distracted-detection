from __future__ import annotations

import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ModelExporter:
    @staticmethod
    def export_to_onnx(
        pt_model_path: str | Path,
        output_path: str | Path | None = None,
        imgsz: tuple[int, int] = (320, 320),
        half: bool = True,
    ) -> Path:
        from ultralytics import YOLO

        model = YOLO(str(pt_model_path), task="detect")
        exported = model.export(format="onnx", half=half, imgsz=imgsz)
        exported_path = Path(exported)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported_path), str(output_path))
            exported_path = output_path
        logger.info("model_exported_onnx", src=str(pt_model_path), dst=str(exported_path))
        return exported_path

    @staticmethod
    def export_to_openvino(
        pt_model_path: str | Path,
        output_path: str | Path | None = None,
        imgsz: tuple[int, int] = (320, 320),
        half: bool = True,
        dynamic: bool = False,
    ) -> Path:
        from ultralytics import YOLO

        model = YOLO(str(pt_model_path), task="detect")
        exported = model.export(format="openvino", dynamic=dynamic, half=half, imgsz=imgsz)
        exported_path = Path(exported)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if exported_path != output_path:
                shutil.move(str(exported_path), str(output_path))
                exported_path = output_path
        logger.info("model_exported_openvino", src=str(pt_model_path), dst=str(exported_path))
        return exported_path

    @staticmethod
    def export_to_tensorrt(
        pt_model_path: str | Path, output_path: str | Path | None = None, imgsz: tuple[int, int] = (320, 320)
    ) -> Path:
        from ultralytics import YOLO

        model = YOLO(str(pt_model_path), task="detect")
        exported = model.export(format="engine", imgsz=imgsz)
        exported_path = Path(exported)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported_path), str(output_path))
            exported_path = output_path
        logger.info("model_exported_tensorrt", src=str(pt_model_path), dst=str(exported_path))
        return exported_path

    @staticmethod
    def auto_export(
        pt_model_path: str | Path,
        target_format: str,
        output_dir: str | Path = "models",
        imgsz: tuple[int, int] = (320, 320),
    ) -> Path:
        pt_path = Path(pt_model_path)
        stem = pt_path.stem
        output_dir = Path(output_dir)
        exporters = {
            "onnx": (output_dir / "cpu_model" / f"{stem}.onnx", ModelExporter.export_to_onnx),
            "openvino": (output_dir / "openVINO_model" / f"{stem}_openvino_model", ModelExporter.export_to_openvino),
            "tensorrt": (output_dir / "tensorrt_model" / f"{stem}.engine", ModelExporter.export_to_tensorrt),
        }
        if target_format not in exporters:
            msg = f"Unknown format: {target_format}. Available: {list(exporters.keys())}"
            raise ValueError(msg)
        output_path, export_fn = exporters[target_format]
        if output_path.exists():
            logger.info("model_already_exported", format=target_format, path=str(output_path))
            if target_format == "openvino" and output_path.is_dir():
                return output_path / f"{stem}.xml"
            return output_path
        final_path = export_fn(pt_model_path, output_path=output_path, imgsz=imgsz)
        if target_format == "openvino" and final_path.is_dir():
            return final_path / f"{stem}.xml"
        return final_path
