"""Typed data contracts used throughout the VitalSense pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class AnalysisConfig:
    window_seconds: float = 60.0
    step_seconds: float = 30.0
    minimum_window_seconds: float = 40.0
    minimum_fusion_quality: float = 0.50
    minimum_modality_quality: float = 0.35
    static_accel_rms_g: float = 0.035
    static_gyro_rms_dps: float = 5.0
    sleep_minimum_minutes: float = 5.0
    posture_change_degrees: float = 30.0
    active_enmo_mg: float = 35.0
    heart_band_hz: tuple[float, float] = (0.30, 4.0)
    respiration_band_hz: tuple[float, float] = (0.08, 0.80)
    minimum_heart_bpm: float = 20.0
    maximum_heart_bpm: float = 240.0
    minimum_resp_bpm: float = 6.0
    maximum_resp_bpm: float = 48.0
    respiration_agreement_bpm: float = 3.0
    piezo_impulse_artifact_ratio: float = 12.0


@dataclass(slots=True)
class DogProfile:
    dog_id: str = "unknown"
    age_years: float | None = None
    weight_kg: float | None = None
    breed: str | None = None
    sex: str | None = None
    neuter_status: str | None = None
    health_status: str | None = None
    start_time_utc: str | None = None
    timezone: str | None = None


@dataclass(slots=True)
class SensorBatch:
    timestamp_s: np.ndarray
    piezo: np.ndarray
    accel_x_g: np.ndarray
    accel_y_g: np.ndarray
    accel_z_g: np.ndarray
    gyro_x_dps: np.ndarray
    gyro_y_dps: np.ndarray
    gyro_z_dps: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.timestamp_s,
            self.piezo,
            self.accel_x_g,
            self.accel_y_g,
            self.accel_z_g,
            self.gyro_x_dps,
            self.gyro_y_dps,
            self.gyro_z_dps,
        )
        lengths = {len(np.asarray(value)) for value in arrays}
        if len(lengths) != 1:
            raise ValueError("Every sensor channel must have the same number of samples")
        if not lengths or next(iter(lengths)) < 2:
            raise ValueError("A sensor batch needs at least two samples")
        for name in self.__dataclass_fields__:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if not np.all(np.isfinite(self.timestamp_s)):
            raise ValueError("Timestamps must be finite")
        if np.any(np.diff(self.timestamp_s) <= 0):
            raise ValueError("Timestamps must be strictly increasing")

    @property
    def sample_rate_hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.timestamp_s)))

    @property
    def duration_s(self) -> float:
        return float(self.timestamp_s[-1] - self.timestamp_s[0])

    @property
    def acceleration(self) -> np.ndarray:
        return np.column_stack((self.accel_x_g, self.accel_y_g, self.accel_z_g))

    @property
    def gyroscope(self) -> np.ndarray:
        return np.column_stack((self.gyro_x_dps, self.gyro_y_dps, self.gyro_z_dps))

    def slice(self, start: int, stop: int) -> "SensorBatch":
        return SensorBatch(**{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__})


@dataclass(slots=True)
class DetectedEvents:
    times_s: list[float] = field(default_factory=list)
    intervals_s: list[float] = field(default_factory=list)
    rate_bpm: float | None = None
    quality: float = 0.0
    amplitude: float | None = None
    source: str = "none"


@dataclass(slots=True)
class WindowResult:
    start_s: float
    end_s: float
    context: str
    orientation_bin: str
    static_candidate: bool
    panting_candidate: bool
    piezo_quality: float
    imu_quality: float
    fusion_quality: float
    valid: bool
    reason_codes: list[str]
    activity_enmo_mg: float
    activity_intensity: float
    accel_dynamic_rms_g: float
    gyro_rms_dps: float
    gravity_vector_g: list[float]
    heart_rate_bpm: float | None = None
    respiratory_rate_bpm: float | None = None
    piezo_resp_rate_bpm: float | None = None
    imu_resp_rate_bpm: float | None = None
    pulse_count: int = 0
    breath_count: int = 0
    pulse_intervals_s: list[float] = field(default_factory=list)
    breath_intervals_s: list[float] = field(default_factory=list)
    interbeat_metrics: dict[str, float | None] = field(default_factory=dict)
    respiratory_variability: dict[str, float | None] = field(default_factory=dict)
    cardiorespiratory_coupling: dict[str, float | None] = field(default_factory=dict)
    pulse_amplitude: float | None = None
    respiratory_amplitude: float | None = None
    signal_metrics: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
