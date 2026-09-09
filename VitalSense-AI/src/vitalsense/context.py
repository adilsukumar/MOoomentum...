"""BMI270-derived motion, static-state, orientation, and quality features."""

from __future__ import annotations

import numpy as np

from .models import AnalysisConfig, SensorBatch
from .signal import (
    clamp,
    clipping_fraction,
    orientation_bin,
    robust_scale,
    safe_highpass,
    spectral_concentration,
    timestamp_coverage,
)


def characterize_window(batch: SensorBatch, config: AnalysisConfig) -> dict[str, object]:
    sample_rate = batch.sample_rate_hz
    gravity = np.nanmedian(batch.acceleration, axis=0)
    dynamic_axes = np.column_stack(
        [safe_highpass(batch.acceleration[:, axis], sample_rate, 0.35) for axis in range(3)]
    )
    dynamic_magnitude = np.linalg.norm(dynamic_axes, axis=1)
    accel_dynamic_rms = float(np.sqrt(np.mean(np.square(dynamic_magnitude))))
    gyro_magnitude = np.linalg.norm(batch.gyroscope, axis=1)
    gyro_rms = float(np.sqrt(np.mean(np.square(gyro_magnitude))))
    raw_accel_magnitude = np.linalg.norm(batch.acceleration, axis=1)
    enmo_mg = float(np.mean(np.maximum(raw_accel_magnitude - 1.0, 0.0)) * 1000.0)
    intensity = float(np.sqrt(np.mean(np.square(np.maximum(raw_accel_magnitude - 1.0, 0.0)))) * 1000.0)
    static = accel_dynamic_rms <= config.static_accel_rms_g and gyro_rms <= config.static_gyro_rms_dps

    timestamp_score, gap_fraction = timestamp_coverage(batch.timestamp_s)
    piezo_clip = clipping_fraction(batch.piezo)
    accel_clip = max(clipping_fraction(batch.acceleration[:, axis]) for axis in range(3))
    heart_concentration = spectral_concentration(batch.piezo, sample_rate, config.heart_band_hz)
    resp_concentration = max(
        spectral_concentration(batch.acceleration[:, axis], sample_rate, config.respiration_band_hz)
        for axis in range(3)
    )
    panting_concentration = max(
        spectral_concentration(batch.acceleration[:, axis], sample_rate, (0.80, 3.0))
        for axis in range(3)
    )
    panting_candidate = bool(
        static and panting_concentration >= 0.45 and panting_concentration > resp_concentration * 1.5
    )
    piezo_scale = robust_scale(batch.piezo)
    finite_piezo = batch.piezo[np.isfinite(batch.piezo)]
    piezo_peak_to_scale = (
        float(np.max(np.abs(finite_piezo - np.median(finite_piezo))) / max(piezo_scale, 1e-9))
        if finite_piezo.size
        else 0.0
    )
    piezo_impulse_artifact = bool(piezo_peak_to_scale >= config.piezo_impulse_artifact_ratio)
    piezo_presence = clamp(piezo_scale / max(np.std(batch.piezo), 1e-9))
    piezo_quality = clamp(
        0.30 * timestamp_score
        + 0.25 * (1.0 - min(piezo_clip * 10.0, 1.0))
        + 0.30 * heart_concentration
        + 0.15 * piezo_presence
    )
    if piezo_impulse_artifact:
        piezo_quality = min(piezo_quality, config.minimum_modality_quality * 0.5)
    motion_suitability = clamp(1.0 - accel_dynamic_rms / max(config.static_accel_rms_g * 3.0, 1e-9))
    imu_quality = clamp(
        0.30 * timestamp_score
        + 0.20 * (1.0 - min(accel_clip * 10.0, 1.0))
        + 0.30 * resp_concentration
        + 0.20 * motion_suitability
    )

    reason_codes: list[str] = []
    if gap_fraction > 0.01:
        reason_codes.append("timestamp_gaps")
    if piezo_clip > 0.01:
        reason_codes.append("piezo_clipping_or_flatline")
    if accel_clip > 0.01:
        reason_codes.append("imu_clipping_or_flatline")
    if not static:
        reason_codes.append("motion_artifact")
    if panting_candidate:
        reason_codes.append("panting_or_high_frequency_breathing_candidate")
    if piezo_impulse_artifact:
        reason_codes.append("piezo_impulse_artifact_candidate")
    if piezo_scale <= 1e-9:
        reason_codes.append("piezo_contact_loss")

    return {
        "gravity_vector_g": [float(value) for value in gravity],
        "orientation_bin": orientation_bin(gravity),
        "accel_dynamic_rms_g": accel_dynamic_rms,
        "gyro_rms_dps": gyro_rms,
        "activity_enmo_mg": enmo_mg,
        "activity_intensity": intensity,
        "static_candidate": static,
        "panting_candidate": panting_candidate,
        "piezo_impulse_artifact_candidate": piezo_impulse_artifact,
        "piezo_quality": piezo_quality,
        "imu_quality": imu_quality,
        "reason_codes": reason_codes,
        "signal_metrics": {
            "timestamp_coverage": timestamp_score,
            "timestamp_gap_fraction": gap_fraction,
            "piezo_clipping_fraction": piezo_clip,
            "imu_clipping_fraction": accel_clip,
            "piezo_heart_spectral_concentration": heart_concentration,
            "piezo_peak_to_robust_scale_ratio": piezo_peak_to_scale,
            "imu_resp_spectral_concentration": resp_concentration,
            "imu_panting_spectral_concentration": panting_concentration,
        },
    }
