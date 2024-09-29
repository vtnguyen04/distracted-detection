import cv2
import torch
import numpy as np
from ultralytics import YOLO
import constant
import ipywidgets as widgets
import openvino as ov
from typing import Any, Dict
from xml.etree.ElementTree import Element
from multiprocessing import Value, Array, Event
import mediapipe as mp

class predict:
    def __init__(
        self, 
        xml_path: Element, 
        device: str
    ):
      
        self.xml_path = xml_path
        self.class_names: Dict[int, str] = {
            0: 'Face', 
            1: 'Eye open', 
            2: 'Eye closed', 
            3: 'Eye open', 
            4: 'Eye closed', 
            5: 'Mouth'
        }
        
        self.CONFIDENCE_THRESHOLD: float = 0.5
        self.prev_frame_time: int = 0
        self.eye_closed_detect: int = 0
        self.model_path = None

        self.face_device: str = device

        if self.face_device == constant.CUDA:
            self.model_path = xml_path.find('face_detect_model_Path').find('cuda_model_Path').text
        else:
            self.model_path = xml_path.find('face_detect_model_Path').find('cpu_model_Path').text


        self.mp_face_mesh = None
        self.face_mesh = None

        self.MEDIAPIPE_TO_DLIB = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454,
            389, 71, 63, 105, 66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97,
            2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
            61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87
        ]
       

    def run(
        self,
        running: Value, 
        show_event: Event, 
        new_frame_event: Event, 
        cropped_frame_np: Array, 
        smemory_results: Array, 
        # smemory_mesh_results: Array,
        smemory_face_detected: Value,
    ) -> None: 
        
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode = False,
                max_num_faces = 1,
                refine_landmarks = True,
                min_detection_confidence = 0.5,
                min_tracking_confidence = 0.5
            )

            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection = 1, 
                min_detection_confidence = 0.5
            )
            
            det_model = None
            if self.face_device == constant.CUDA:
                det_model = YOLO(
                    self.xml_path.find('drowsy_detect_model_Path').find('cuda_model_Path').text,
                    task = "detect"
                )

            elif self.face_device == constant.LOCAL:

                det_model = YOLO(self.xml_path.find('default_model_Path').find('face').text)
                det_model()
                det_model_path = constant.DET_MODEL_PATH

                core = ov.Core()

                device = widgets.Dropdown(
                    options = core.available_devices + ["AUTO"],
                    value = "AUTO",
                    description = "Device:",
                    disabled = False,
                )

                det_ov_model = core.read_model(det_model_path)
                ov_config = {}

                if device.value != "CPU":
                    det_ov_model.reshape(
                        {0: [1, 3, constant.CROP_HEIGHT, constant.CROP_WIDTH]}
                    )

                if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
                    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES", "GPU_HOST_TASK_PRIORITY": "HIGH"}

                compiled_model = core.compile_model(det_ov_model, device.value, ov_config)

                det_model.predictor.inference = lambda *args: torch.from_numpy(compiled_model(args)[0])
                det_model.predictor.model.pt = False

            cv2.ocl.setUseOpenCL(True)
            cv2.setUseOptimized(True)

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, constant.TARGET_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, constant.TARGET_HEIGHT)

            x_min, y_min, x_max, y_max = 0, 0, 0, 0

            while cap.isOpened():
                ret, frame_resized = cap.read()
                if not ret:
                    break

                new_frame_event.set()
                if smemory_face_detected.value == constant.FALSE:

                    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                    results = self.face_detection.process(image_rgb)

                    if results.detections:
                        for detection in results.detections:
                            smemory_face_detected.value = constant.TRUE
                            bboxC = detection.location_data.relative_bounding_box
                            h, w, _ = frame_resized.shape

                            center_x = int((bboxC.xmin + bboxC.width / 2) * w)
                            center_y = int((bboxC.ymin + bboxC.height / 2) * h)

                            x_min = center_x - constant.CROP_SIZE_HALF
                            y_min = center_y - constant.CROP_SIZE_HALF
                            x_max = x_min + constant.CROP_WIDTH
                            y_max = y_min + constant.CROP_HEIGHT

                            if x_max < constant.CROP_WIDTH or y_max < constant.CROP_HEIGHT:
                                smemory_face_detected.value = constant.FALSE
                            elif x_max > constant.TARGET_WIDTH or y_max > constant.TARGET_HEIGHT:
                                smemory_face_detected.value = constant.FALSE


                elif smemory_face_detected.value == constant.TRUE:

                    arr = np.frombuffer(
                        cropped_frame_np.buf, 
                        dtype = np.uint8
                    ).reshape(constant.input_shape)
                    
                    arr[:] = np.resize(
                        frame_resized[y_min: y_max, x_min: x_max, :], 
                        constant.input_shape
                    )

                    resized_frame_torch = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255
                
                    face_results = det_model(resized_frame_torch, max_det = 4)

                    # mesh_results = self.process_frame(frame_resized[y_min:y_max, x_min:x_max, :])

                    # res_mesh_np_arr = np.ndarray(
                    #     buffer = smemory_mesh_results.buf, 
                    #     dtype = np.float16, 
                    #     shape = constant.mesh_result_shape
                    # )


                    # res_mesh_np_arr[: mesh_results.shape[0], :] = mesh_results
                    for result in face_results:
                        res_np_arr = np.ndarray(
                            buffer = smemory_results.buf, 
                            dtype = np.float16, 
                            shape = constant.result_shape
                        )
                        result_boxes = result.boxes.data.numpy()
                        res_np_arr[:result_boxes.shape[0], :] = result_boxes

                    show_event.set()
                
                if running.value == constant.NOT_RUNNING:
                    break
            cap.release()
            cv2.destroyAllWindows()
        finally:
            if self.face_mesh:
                self.face_mesh.close()
            pass


    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                shape = []
                for idx in self.MEDIAPIPE_TO_DLIB:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    shape.append(np.array([x, y]))
                
                
                shape = np.array(shape)
                return shape
        else:
            pass

        return frame