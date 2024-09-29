import sound_play
import time
import numpy as np
import constant
import cv2
from multiprocessing import Value, Array, Event
from typing import Dict

class detect_process:

    def __init__(
        self, 
        sound_path: str
    ):
        self.sound = sound_play.Sound(sound_path)

    def recur_time_calculator(
        self, 
        fps: Value, 
        new_frame_event: Event, 
        eye_closed_cnt: Value, 
        eye_state: Value, 
        eye_state_timeline: Value,
        frame_cnt: Value, 
        smemory_face_detected: Value
    ) -> None:
     
        prev_frame_time: float = 0
        new_frame_time: float = 0
        copy_eye_state: int = 0
        past_time: float = time.perf_counter()
      
        while True:
            new_frame_event.wait()
            new_frame_time = time.perf_counter()
            fps.value = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time

            if smemory_face_detected.value == constant.TRUE:
                if (time.perf_counter() - past_time) > 2:
                    past_time = time.perf_counter()
                    smemory_face_detected.value = constant.FALSE

            frame_cnt.value += 1

            copy_eye_state = eye_closed_cnt.value
            eye_state.value += copy_eye_state
            eye_state_timeline.value += copy_eye_state

            new_frame_event.clear()

    def eye_state_clock(
        self, 
        smemory_eyeopen: Value, 
        new_frame_event: Event, 
        smemory_is_drowsy: Value, 
        eye_state: Value, 
        frame_cnt: Value,
        eye_state_timeline: Value, 
        smemory_face_detected: Value
    ) -> None:
        
        two_sec_clock: float = time.perf_counter()
        frame_cnt_0_5: int = constant.INIT_VAL
        frame_cnt_2_0: int = constant.INIT_VAL
        
        while True:
            new_frame_event.wait()

            if (time.perf_counter() - two_sec_clock) > 4.5:
                break
            new_frame_event.clear()

        while_count: int = constant.INIT_VAL
        eye_state_timeline.value = constant.INIT_VAL
        
        while True:
            while_count += 1
            frame_cnt_0_5 = frame_cnt.value
            frame_cnt.value = constant.INIT_VAL
            frame_cnt_2_0 += frame_cnt_0_5
            if smemory_face_detected.value == constant.TRUE:
                self.is_Not_Drowsy(smemory_eyeopen, frame_cnt_0_5, smemory_is_drowsy)

            if while_count == 4:
                while_count = constant.INIT_VAL
                if smemory_face_detected.value == constant.TRUE:
                    self.is_Drowsy(eye_state, frame_cnt_2_0, smemory_is_drowsy)
                eye_state.value -= eye_state_timeline.value
                eye_state_timeline.value = constant.INIT_VAL
                frame_cnt_2_0 = constant.INIT_VAL
            time.sleep(0.5)

    def is_Not_Drowsy(
        self, 
        smemory_eyeopen: Value, 
        frame_cnt_0_5: int, 
        smemory_is_drowsy: Value
    ) -> None:
        if frame_cnt_0_5 != 0:
            if (smemory_eyeopen.value / frame_cnt_0_5) < constant.EYE_OPEN_RATE_FPS:
                smemory_is_drowsy.value = constant.FALSE
                self.awake()
            smemory_eyeopen.value = constant.INIT_VAL

    def is_Drowsy(
        self, 
        eye_state: Value, 
        frame_cnt_2_0: int, 
        smemory_is_drowsy: Value
    ) -> None:
        
        if frame_cnt_2_0 != 0:
            if (eye_state.value / frame_cnt_2_0) > constant.EYE_CLOSED_RATE_FPS:
                smemory_is_drowsy.value = constant.TRUE
                self.drowsy()

    def drowsy(
        self
    ) -> None:
        if not self.sound.is_playing():
            self.sound.warn()

    def awake(
        self
    ) -> None:
        if self.sound.is_playing():
            self.sound.warn_stop()

    def image_show(
        self, 
        smemory_results: Array, 
        # smemory_mesh_results: Array, 
        show_event: Event, 
        smemory_eyeclosed: Value, 
        smemory_eyeopen: Value, 
        cropped_frame_np: Array, 
        smemory_is_drowsy: Value, 
        smemory_fps: Value
    ) -> None:
    
        class_names: Dict[int, str] = {
            0: 'Face', 
            1: 'Eye open', 
            2: 'Eye closed', 
            3: 'Eye open', 
            4: 'Eye closed', 
            5: 'Mouth'
        }

        eye_closed_detect: float = 0
        show_event.wait()
        show_event.clear()

        while True:
            show_event.wait()
            np_crop = np.frombuffer(
                cropped_frame_np.buf, 
                dtype = np.uint8
            ).reshape(constant.input_shape).copy()
            c = np_crop.copy()
            result_copy = np.ndarray(
                buffer = smemory_results.buf, 
                dtype = np.float16, 
                shape = constant.result_shape
            ).copy()

            # messh_result_copy = np.ndarray(
            #     buffer = smemory_mesh_results.buf, 
            #     dtype = np.float16, 
            #     shape = constant.mesh_result_shape
            # ).copy()
            
            # for (x, y) in messh_result_copy:
            #     cv2.circle(c, (int(x), int(y)), 1, (0, 0, 255), -1)

            for box in result_copy:
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box[4]
                cls = int(box[5])

                if conf < 0.3:
                    continue

                label = class_names.get(cls, 'Unknown')
                color = (255, 255, 255)

                if label == 'Eye closed':
                    eye_closed_detect += 0.5
                    color = (0, 0, 255)
                elif label == 'Eye open':
                    smemory_eyeopen.value -= 0.5
                    color = (0, 255, 0)

                # Draw rectangle and label with adjusted font size and thickness
                cv2.rectangle(np_crop, (x1, y1), (x2, y2), color, 2)
                cv2.putText(np_crop, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)  # Adjusted font size to 0.6, thickness to 1

            smemory_eyeclosed.value = eye_closed_detect
            eye_closed_detect = 0

            if smemory_is_drowsy.value == constant.TRUE:
                cv2.putText(np_crop, 'Drowsiness!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Adjusted size and thickness
            cv2.putText(np_crop, f'FPS: {smemory_fps.value}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Adjusted size and thickness
            cv2.imshow('Drowsiness Detection', np_crop)
            # cv2.imshow('Drowsiness Detection2', c)
            cv2.waitKey(1)
            show_event.clear()

