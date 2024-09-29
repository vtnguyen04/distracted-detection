import time
from playsound import playsound  # Thay thế winsound bằng playsound
import constant

class Sound:
    def __init__(
        self, 
        sound_path: str
    ):
       
        self.begin_time = 0
        self.PLAY_TIME = constant.SOUND_PLAY_SEC
        self.is_stopped = False
        self.sound_path = sound_path

    def is_playing(self) -> bool:
        if self.is_stopped:
            return False
        return (time.perf_counter() - self.begin_time) < self.PLAY_TIME

    def warn(
        self
    ) -> None:
        self.is_stopped = False
        self.begin_time = time.perf_counter()
        playsound(self.sound_path)  # Chạy âm thanh

    def warn_stop(
        self
    ) -> None:
        self.is_stopped = True
        # playsound không có phương thức dừng như winsound
        # nên chỉ cần điều chỉnh trạng thái
