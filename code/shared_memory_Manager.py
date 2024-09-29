import multiprocessing
import numpy as np
import constant
from multiprocessing import shared_memory
from typing import Dict, Any


class SharedMemoryManager:
    def __init__(
        self
    ) -> None:
        
        self.shared_memory: Dict[str, Any] = {

            # Check if the program is running
            'running': multiprocessing.Value('i', 0),

            # Frame processing + event
            'fps': multiprocessing.Value('i', 0),
            'new_frame_event': multiprocessing.Event(),
            'show_event': multiprocessing.Event(),
            'frame_cnt': multiprocessing.Value('i', 0),

            # State of eyes
            'eye_closed_cnt': multiprocessing.Value('f', 0.0),
            'eye_open_cnt': multiprocessing.Value('f', 0.0),
            'is_drowsy': multiprocessing.Value('i', 0),
            'eye_state': multiprocessing.Value('f', 0.0),
            'eye_state_timeline': multiprocessing.Value('f', 0),

            # Memory to save cropped frame
            'cropped_frame_np': shared_memory.SharedMemory(
                create = True, 
                size = int(np.prod(constant.input_shape))
            ),

            # Result of prediction & result of face detection
            'smemory_results': shared_memory.SharedMemory(
                create = True, 
                size = int(
                    np.prod(constant.result_shape) * np.dtype(np.float16).itemsize
                )
            ),

            'smemory_mesh_results': shared_memory.SharedMemory(
                create = True, 
                size = int(
                    np.prod(constant.mesh_result_shape) * np.dtype(np.float16).itemsize
                )
            ),

            'smemory_face_detected': multiprocessing.Value('i', 0),
        }

    def get_memory(
        self, 
        *keys: str
    ) -> Dict[str, Any]:
        return {
            key: self.shared_memory[key] for key in keys
        }

    def set_memory(
        self, 
        key: str, 
        value: Any
    ) -> None:
        
        self.shared_memory[key].value = value

    def get_value(
        self, 
        key: str
    ) -> Any:
        return self.shared_memory[key].value

    def kill_process(
        self
    ) -> None:
        
        for name, shm in self.shared_memory.items():
            if isinstance(shm, shared_memory.SharedMemory):
                shm.close()
                shm.unlink()


