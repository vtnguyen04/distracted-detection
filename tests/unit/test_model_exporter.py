from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.infrastructure.model_exporter import ModelExporter


@pytest.fixture
def mock_yolo_pt(tmp_path: Path) -> Path:
    """Create a dummy .pt file to act as input model."""
    pt_file = tmp_path / "dummy_model.pt"
    pt_file.touch()
    return pt_file


@patch("src.infrastructure.model_exporter.ModelExporter.export_to_onnx")
def test_auto_export_onnx(mock_export: MagicMock, mock_yolo_pt: Path, tmp_path: Path) -> None:
    """Test auto_export routes to ONNX properly."""
    output_dir = tmp_path / "models"
    ModelExporter.auto_export(mock_yolo_pt, "onnx", output_dir=output_dir)
    expected_output = output_dir / "cpu_model" / "dummy_model.onnx"
    mock_export.assert_called_once_with(mock_yolo_pt, output_path=expected_output, imgsz=(320, 320))


@patch("src.infrastructure.model_exporter.ModelExporter.export_to_openvino")
def test_auto_export_openvino(mock_export: MagicMock, mock_yolo_pt: Path, tmp_path: Path) -> None:
    """Test auto_export routes to OpenVINO properly."""
    output_dir = tmp_path / "models"
    ModelExporter.auto_export(mock_yolo_pt, "openvino", output_dir=output_dir)
    expected_output = output_dir / "openVINO_model" / "dummy_model_openvino_model"
    mock_export.assert_called_once_with(mock_yolo_pt, output_path=expected_output, imgsz=(320, 320))


def test_auto_export_unknown_format(mock_yolo_pt: Path) -> None:
    """Test auto_export raises ValueError for unknown format."""
    with pytest.raises(ValueError):
        ModelExporter.auto_export(mock_yolo_pt, "unknown_format")


def test_auto_export_skips_if_exists(mock_yolo_pt: Path, tmp_path: Path) -> None:
    """Test auto_export returns early if target already exists."""
    output_dir = tmp_path / "models"
    target_path = output_dir / "cpu_model" / "dummy_model.onnx"
    target_path.parent.mkdir(parents=True)
    target_path.touch()
    with patch("src.infrastructure.model_exporter.ModelExporter.export_to_onnx") as mock_export:
        result = ModelExporter.auto_export(mock_yolo_pt, "onnx", output_dir=output_dir)
        assert result == target_path
        mock_export.assert_not_called()
