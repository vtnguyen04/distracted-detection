import time

import pytest
from src.config.settings import AlertSettings
from src.domain.enums import AlertLevel
from src.infrastructure.alert_manager import AlertManager


@pytest.fixture
def alert_manager():
    return AlertManager(AlertSettings(enabled=True))


def test_alert_manager_init(alert_manager):
    assert alert_manager._settings.enabled is True


def test_trigger_alert_disabled():
    manager = AlertManager(AlertSettings(enabled=False))
    manager.play_alert(AlertLevel.HIGH)
    assert manager._current_level == AlertLevel.NONE


def test_trigger_alert_plays_sound(alert_manager, mocker):
    mocker.patch("shutil.which", return_value="/bin/dummy_player")
    mock_popen = mocker.patch("subprocess.Popen")

    alert_manager._audio_cache = {AlertLevel.HIGH: "/path/to/sound.wav"}
    alert_manager.play_alert(AlertLevel.HIGH)

    time.sleep(0.1)  # Yield thread

    assert alert_manager._current_level == AlertLevel.HIGH
    mock_popen.assert_called()


def test_stop_alert_stops_sound(alert_manager, mocker):
    alert_manager._current_level = AlertLevel.HIGH
    alert_manager.stop_alert()
    assert alert_manager._current_level == AlertLevel.NONE
