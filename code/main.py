import multiprocessing
import time
import sys
from typing import Type, Dict
import constant
import detection
import gui_manager
import output_predict
import xml.etree.ElementTree as ET
import shared_memory_Manager
import model_exporter
import platform

if platform.system() == 'Windows':
    import winsound
else:
    from playsound import playsound


class ProcessManager:

    def __init__(
        self
    ) -> None:
        
        self.processes: Dict[str, Type[multiprocessing.Process]] = {}
        self.s_memory: shared_memory_Manager.SharedMemoryManager = None
        self.predict_process: output_predict.predict = None
        self.detect_process: detection.detect_process = None
        # self.gaze_frame_queue = multiprocessing.Queue() 
        self.device: str = constant.LOCAL

        
    def init_program(
        self
    ) -> None:
       
        path_tree = ET.parse('paths.xml').getroot()
        sound_path: str = path_tree.find('sound_Path').text
        model_path: str = path_tree.find('model_path')

        # model_exporter.exporter(model_path, self.device)

        self.detect_process = detection.detect_process(sound_path)
        # self.gaze_detect_process = detection.gaze_detect_process()
        self.predict_process = output_predict.predict(model_path, self.device)
        self.s_memory = shared_memory_Manager.SharedMemoryManager()

        # tạo các tiến trình của dự án
        self.processes = {
            'eye_state_clock': Type[multiprocessing.Process],
            'detect': Type[multiprocessing.Process],
            'predict': Type[multiprocessing.Process],
            'image_show': Type[multiprocessing.Process],
            # 'gaze_detect': Type[multiprocessing.Process]
        }

        self.s_memory.set_memory('running', constant.NOT_RUNNING)

    def start_processes(
        self
    ) -> None:

        if self.s_memory.get_value('running') == constant.NOT_RUNNING:
            try:
                self.s_memory.set_memory('running', constant.RUNNING)

                self.processes['image_show'] = multiprocessing.Process(
                    target = self.detect_process.image_show,
                    args = (
                        *self.s_memory.get_memory(
                            'smemory_results', 
                            # 'smemory_mesh_results',
                            'show_event', 
                            'eye_closed_cnt', 
                            'eye_open_cnt',
                            'cropped_frame_np', 
                            'is_drowsy', 
                            'fps'
                        ).values(),
                    ),
                )

                self.processes['eye_state_clock'] = multiprocessing.Process(
                    target = self.detect_process.eye_state_clock,
                    args = (
                        *self.s_memory.get_memory(
                            'eye_open_cnt', 
                            'new_frame_event', 
                            'is_drowsy', 
                            'eye_state',
                            'frame_cnt', 
                            'eye_state_timeline', 
                            'smemory_face_detected'
                        ).values(),
                    ),
                )

                self.processes['detect'] = multiprocessing.Process(
                    target = self.detect_process.recur_time_calculator,
                    args = (
                        *self.s_memory.get_memory(
                            'fps', 
                            'new_frame_event', 
                            'eye_closed_cnt', 
                            'eye_state',
                            'eye_state_timeline', 
                            'frame_cnt', 
                            'smemory_face_detected'
                        ).values(),
                    ),
                )
                self.processes['predict'] = multiprocessing.Process(
                    target = self.predict_process.run,
                    args = (
                        *self.s_memory.get_memory(
                            'running', 
                            'show_event', 
                            'new_frame_event', 
                            'cropped_frame_np',
                            'smemory_results',
                            # 'smemory_mesh_results',
                            'smemory_face_detected'
                        ).values(),
                    ),
                )

                # self.processes['gaze_detect'] = multiprocessing.Process(
                #     target = self.gaze_detect_process.run_gaze_detection,
                #     args = (self.gaze_frame_queue,)  # Pass the frame queue to the gaze detect process
                # )
                    
                # self.processes['gaze_detect'] = multiprocessing.Process(
                #     target=self.gaze_detect_process.run_gaze_detection  # Run the gaze detection process
                # )

                for process in self.processes.values():
                    process.start()

            except multiprocessing.ProcessError as err:
                self.stop_processes()
                print(f"Process error: {err}")
            
            except Exception as err:
                self.stop_processes()
                print(f"Unexpected error: {err}")


    def stop_processes(
        self
    ) -> None:
        
        if self.s_memory.get_value('running') == constant.RUNNING:
            self.s_memory.set_memory('running', constant.NOT_RUNNING)
            time.sleep(1)

            if platform.system() == 'Windows':
                winsound.PlaySound(None, winsound.SND_PURGE)

            for process in self.processes.values():
                if process is None:
                    continue
                else:
                    if process.is_alive():
                        process.terminate()

            self.s_memory.kill_process()


if __name__ == '__main__':
    manager = ProcessManager()
    manager.init_program()
    window = gui_manager.manager()
    window.start_window(manager)
