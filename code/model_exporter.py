import os
import shutil
from typing import Optional
from xml.etree.ElementTree import Element
from ultralytics import YOLO
import constant

class exporter:
    def __init__(
        self, 
        model_path: Element, 
        device: str
    ) -> None:
     
        self.base_path: str = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.face_pt_path: str = self._get_model_path(
            model_path, 
            'default_model_Path', 
            'face'
        )
        self.drowsy_pt_path: str = self._get_model_path(
            model_path, 
            'default_model_Path', 
            'drowsy'
        )
        self.device: str = device

        if self.device == 'cuda':
            self.export_to_cuda(model_path)
        else:
            self.export_to_onnx_and_openvino(model_path)

    def _get_model_path(
        self, 
        model_path: Element, 
        base_tag: str, 
        sub_tag: str
    ) -> str:
        """Helper method to get the model path from XML."""
     
        return self.base_path + model_path.find(base_tag).find(sub_tag).text.strip()

    def export_to_cuda(
        self, 
        model_path: Element
    ) -> None:
     
        """Exports models to TensorRT (CUDA engine) format."""
     
        cuda_face_model_path: str = self._get_model_path(
            model_path, 
            'face_detect_model_Path', 
            'cuda_model_Path'
        )
        cuda_detection_model_path: str = self._get_model_path(
            model_path, 
            'drowsy_detect_model_Path', 
            'cuda_model_Path'
        )

        if not os.path.exists(cuda_face_model_path):
            self._export_model_to_engine(
                self.face_pt_path, 
                cuda_face_model_path, 
                constant.TARGET_HEIGHT,
                constant.TARGET_WIDTH
            )

        if not os.path.exists(cuda_detection_model_path):
            self._export_model_to_engine(
                self.drowsy_pt_path, 
                cuda_detection_model_path, 
                constant.CROP_HEIGHT, 
                constant.CROP_WIDTH
            )

    def export_to_onnx_and_openvino(
        self, 
        model_path: Element
    ) -> None:
      
        """Exports models to ONNX and OpenVINO format."""
        onnx_face_model_path: str = self._get_model_path(
            model_path, 
            'face_detect_model_Path', 
            'cpu_model_Path'
        )

        openvino_detection_model_path: str = self._get_model_path(
            model_path, 
            'drowsy_detect_model_Path', 
            'openvino_model_Path'
        )

        if not os.path.exists(onnx_face_model_path):
            self._export_model_to_onnx(
                self.face_pt_path, 
                onnx_face_model_path, 
                constant.TARGET_HEIGHT, 
                constant.TARGET_WIDTH
            )

        if not os.path.exists(openvino_detection_model_path):
            self._export_model_to_openvino(
                self.drowsy_pt_path, 
                openvino_detection_model_path, 
                constant.CROP_HEIGHT, 
                constant.CROP_WIDTH
            )

    def _export_model_to_engine(
        self, 
        pt_model_path: str, 
        export_model_path: str, 
        height: int, 
        width: int
    ) -> None:
        
        """Exports the PyTorch model to TensorRT engine format."""
        model_out_path: str = YOLO(
            pt_model_path, 
            task = "detect"
        ).export(
            format = 'engine', 
            imgsz = (height, width)
        )
        
        self._move_model_files(
            model_out_path, 
            export_model_path
        )

    def _export_model_to_onnx(
        self, 
        pt_model_path: str, 
        export_model_path: str, 
        height: int, 
        width: int
    ) -> None:
        """Exports the PyTorch model to ONNX format."""
     
        model_out_path: str = YOLO(
            pt_model_path, 
            task = "detect"
        ).export(
            format = 'onnx', 
            half = True, 
            imgsz = (height, width)
        )

        self._move_model_files(
            model_out_path, 
            export_model_path
        )

        # Optional testing after export
        test_model = YOLO(export_model_path)
        test_model()

    def _export_model_to_openvino(
        self, 
        pt_model_path: str, 
        export_model_path: str, 
        height: int, 
        width: int
    ) -> None:
        
        """Exports the PyTorch model to OpenVINO format."""
        model_out_path: str = YOLO(
            pt_model_path,  
            task = "detect"
        ).export(
            format = 'openvino', 
            dynamic = False, 
            half = True, 
            imgsz = (height, width)
        )
        self._move_model_files(
            model_out_path, 
            export_model_path
        )

    def _move_model_files(
        self, 
        src_path: str, 
        dst_path: str
    ) -> None:
     
        """Moves model files from the source path to the destination path."""
        os.makedirs(
            os.path.dirname(dst_path), 
            exist_ok = True
        )
        
        shutil.move(src_path, dst_path)
