from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.domain.enums import AlertLevel

if TYPE_CHECKING:
    from src.config.settings import AlertSettings
logger = structlog.get_logger()


class AlertManager:
    def __init__(self, settings: AlertSettings) -> None:
        self._settings = settings
        self._is_playing = False
        self._play_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_level: AlertLevel = AlertLevel.NONE
        self._audio_cache: dict[AlertLevel, tuple[np.ndarray, int]] = {}
        if settings.enabled:
            self._preload_sounds()

    def _preload_sounds(self) -> None:
        sound_dir = self._settings.sound_dir
        warning_path = sound_dir / self._settings.warning_sound
        critical_path = sound_dir / self._settings.critical_sound
        if warning_path.exists():
            self._audio_cache[AlertLevel.MEDIUM] = str(warning_path)
            self._audio_cache[AlertLevel.LOW] = str(warning_path)
        if critical_path.exists():
            self._audio_cache[AlertLevel.HIGH] = str(critical_path)
        elif AlertLevel.MEDIUM in self._audio_cache:
            self._audio_cache[AlertLevel.HIGH] = self._audio_cache[AlertLevel.MEDIUM]
        logger.info("audio_preloaded", levels=list(self._audio_cache.keys()), sound_dir=str(sound_dir))

    def play_alert(self, level: AlertLevel) -> None:
        if not self._settings.enabled:
            return
        if self._is_playing and self._current_level == level:
            return
        if level not in self._audio_cache:
            logger.warning("audio_level_not_loaded", level=level.name)
            return
        self.stop_alert()
        self._stop_event.clear()
        self._current_level = level
        self._is_playing = True
        self._play_thread = threading.Thread(target=self._play_loop, args=(level,), daemon=True)
        self._play_thread.start()

    def _play_loop(self, level: AlertLevel) -> None:
        import shutil
        import subprocess
        import time

        audio_path = self._audio_cache[level]
        player = shutil.which("paplay") or shutil.which("aplay")
        if not player:
            self._settings.enabled = False
            logger.warning("audio_playback_error_disabled_backend", error="No native ALSA/PulseAudio player found")
            self._is_playing = False
            return
        try:
            while not self._stop_event.is_set():
                proc = subprocess.Popen([player, audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                while proc.poll() is None:
                    if self._stop_event.is_set():
                        proc.terminate()
                        break
                    time.sleep(0.1)
        except Exception as e:
            self._settings.enabled = False
            logger.warning("audio_playback_error_disabled_backend", error=str(e))
        finally:
            self._is_playing = False

    def stop_alert(self) -> None:
        self._stop_event.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self._is_playing = False
        self._current_level = AlertLevel.NONE

    @property
    def is_playing(self) -> bool:
        return self._is_playing
