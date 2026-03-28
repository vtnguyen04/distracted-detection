from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_CAMERA_")
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    source: str = ""
    retry_attempts: int = 3
    retry_delay_sec: float = 1.0


class DetectorWeights(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_WEIGHT_")
    ear: float = 0.4
    mar: float = 0.1
    head_pose: float = 0.15
    perclos: float = 0.3
    blink_rate: float = 0.05
    yolo_eye: float = 0.35


class EarSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_EAR_")
    threshold: float = 0.15
    consecutive_frames: int = 15


class MarSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_MAR_")
    threshold: float = 0.85
    min_duration_sec: float = 1.5


class HeadPoseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_HEAD_")
    pitch_threshold: float = 20.0
    yaw_threshold: float = 30.0


class PerclosSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_PERCLOS_")
    window_sec: float = 30.0
    threshold: float = 0.45


class BlinkRateSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_BLINK_")
    normal_low: int = 8
    normal_high: int = 35
    window_sec: float = 60.0


class StateMachineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_STATE_")
    warning_threshold: float = 0.50
    distracted_threshold: float = 0.70
    safe_threshold: float = 0.35
    warning_sustain_sec: float = 1.0
    distracted_sustain_sec: float = 1.5
    recovery_sustain_sec: float = 1.5


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_ALERT_")
    sound_dir: Path = Path("assets/audio")
    warning_sound: str = "alert.wav"
    critical_sound: str = "alarm.wav"
    enabled: bool = True


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_INFERENCE_")
    backend: str = "onnx"
    face_model_path: str = "models/cpu_model/v8n_facedetect_model.onnx"
    distracted_model_path: str = "models/pt_model/v8m_distracted_detect_model.pt"
    confidence_threshold: float = 0.4
    device: str = "AUTO"
    crop_size: int = 320


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTRACTED_")
    enabled_detectors: list[str] = Field(default=["ear", "mar", "head_pose", "perclos", "blink_rate", "yolo_eye"])
    model_backend: str = "mediapipe"
    frame_skip: int = 0
    show_ui: bool = True
    log_level: str = "INFO"
    profiling: bool = False
    use_multiprocessing: bool = True


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DISTRACTED_", env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )
    camera: CameraSettings = Field(default_factory=CameraSettings)
    weights: DetectorWeights = Field(default_factory=DetectorWeights)
    ear: EarSettings = Field(default_factory=EarSettings)
    mar: MarSettings = Field(default_factory=MarSettings)
    head_pose: HeadPoseSettings = Field(default_factory=HeadPoseSettings)
    perclos: PerclosSettings = Field(default_factory=PerclosSettings)
    blink_rate: BlinkRateSettings = Field(default_factory=BlinkRateSettings)
    state_machine: StateMachineSettings = Field(default_factory=StateMachineSettings)
    alert: AlertSettings = Field(default_factory=AlertSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        config_path = data.pop("config_path", None)
        if config_path and Path(config_path).exists():
            with Path(config_path).open() as f:
                yaml_data = yaml.safe_load(f) or {}
            for key, value in yaml_data.items():
                if key not in data:
                    data[key] = value
        return data


def load_settings(config_path: str | None = None) -> AppSettings:
    init_data: dict[str, Any] = {}
    if config_path:
        init_data["config_path"] = config_path
    return AppSettings(**init_data)
