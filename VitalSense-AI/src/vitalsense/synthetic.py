"""Deterministic synthetic Piezo + BMI270 data for integration testing."""

from __future__ import annotations

import numpy as np

from .models import SensorBatch


def generate_synthetic_session(
    duration_s: float = 20.0 * 60.0,
    sample_rate_hz: float = 100.0,
    heart_rate_bpm: float = 72.0,
    respiratory_rate_bpm: float = 15.0,
    seed: int = 7,
    include_activity: bool = True,
) -> SensorBatch:
    rng = np.random.default_rng(seed)
    timestamp = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)

    instantaneous_hr = np.full(len(timestamp), heart_rate_bpm, dtype=float)
    instantaneous_rr = np.full(len(timestamp), respiratory_rate_bpm, dtype=float)
    if include_activity and duration_s >= 900.0:
        active_mask = (timestamp >= 360.0) & (timestamp < 510.0)
        recovery_mask = timestamp >= 510.0
        instantaneous_hr[active_mask] += 60.0
        instantaneous_rr[active_mask] += 25.0
        instantaneous_hr[recovery_mask] += 45.0 * np.exp(-(timestamp[recovery_mask] - 510.0) / 150.0)
        instantaneous_rr[recovery_mask] += 14.0 * np.exp(-(timestamp[recovery_mask] - 510.0) / 120.0)

    heart_cycles = np.cumsum(instantaneous_hr / 60.0 / sample_rate_hz)
    beat_indices = np.flatnonzero(np.diff(np.floor(heart_cycles), prepend=np.floor(heart_cycles[0])) > 0)
    beat_times = timestamp[beat_indices]
    piezo = 0.03 * rng.normal(size=len(timestamp))
    for beat_time in beat_times:
        piezo += 1.15 * np.exp(-0.5 * ((timestamp - beat_time) / 0.035) ** 2)
        piezo -= 0.32 * np.exp(-0.5 * ((timestamp - beat_time - 0.085) / 0.050) ** 2)

    respiration_phase = 2.0 * np.pi * np.cumsum(instantaneous_rr / 60.0 / sample_rate_hz)
    respiration = np.sin(respiration_phase)
    piezo += 0.34 * respiration
    accel_x = 0.003 * rng.normal(size=len(timestamp)) + 0.010 * respiration
    accel_y = 0.003 * rng.normal(size=len(timestamp)) + 0.006 * np.sin(respiration_phase + 0.3)
    accel_z = 1.0 + 0.003 * rng.normal(size=len(timestamp)) + 0.018 * respiration
    gyro_x = 0.08 * rng.normal(size=len(timestamp)) + 0.15 * respiration
    gyro_y = 0.08 * rng.normal(size=len(timestamp))
    gyro_z = 0.08 * rng.normal(size=len(timestamp))

    if include_activity and duration_s >= 900.0:
        # Two minutes of movement followed by a long recovery/rest segment.
        active = (timestamp >= 360.0) & (timestamp < 510.0)
        movement = 0.15 * np.sin(2.0 * np.pi * 2.2 * timestamp[active]) + 0.08 * rng.normal(size=np.sum(active))
        accel_x[active] += movement
        accel_y[active] += 0.7 * movement
        accel_z[active] += 0.5 * movement
        gyro_x[active] += 22.0 * np.sin(2.0 * np.pi * 1.4 * timestamp[active])
        gyro_y[active] += 12.0 * rng.normal(size=np.sum(active))
        piezo[active] += 0.8 * rng.normal(size=np.sum(active))

        # A posture change after activity.
        changed = timestamp >= 510.0
        accel_x[changed] += 0.78
        accel_z[changed] -= 0.38

    return SensorBatch(
        timestamp_s=timestamp,
        piezo=piezo,
        accel_x_g=accel_x,
        accel_y_g=accel_y,
        accel_z_g=accel_z,
        gyro_x_dps=gyro_x,
        gyro_y_dps=gyro_y,
        gyro_z_dps=gyro_z,
    )
