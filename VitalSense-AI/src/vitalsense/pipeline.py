"""End-to-end windowed VitalSense analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from .alerts import evaluate_alerts
from .context import characterize_window
from .detectors import detect_pulses, detect_respiration
from .longitudinal import PersonalBaseline, score_against_baseline, summarize_session
from .metrics import cardiorespiratory_coupling, interval_variability, respiratory_variability
from .models import AnalysisConfig, DogProfile, SensorBatch, WindowResult, _json_ready
from .presentation import build_frontend_payload
from .signal import clamp, joint_amplitude_dropout_runs, principal_component, safe_bandpass


def analyze_session(
    batch: SensorBatch,
    profile: DogProfile | None = None,
    config: AnalysisConfig | None = None,
    baseline: PersonalBaseline | None = None,
) -> dict[str, Any]:
    profile = profile or DogProfile()
    config = config or AnalysisConfig()
    windows = _process_windows(batch, config)
    summary = summarize_session(windows, config)
    baseline_score = score_against_baseline(summary, baseline)
    alerts = evaluate_alerts(windows, baseline_score)
    frontend = build_frontend_payload(summary, windows, profile, baseline_score, alerts, config)
    return _json_ready(
        {
            "schema_version": "vitalsense.analysis.v1",
            "algorithm_version": "0.1.0-research",
            "intended_use": "research and trend screening; not a diagnosis",
            "dog_profile": {
                "dog_id": profile.dog_id,
                "age_years": profile.age_years,
                "weight_kg": profile.weight_kg,
                "breed": profile.breed,
                "sex": profile.sex,
                "neuter_status": profile.neuter_status,
                "health_status": profile.health_status,
                "start_time_utc": profile.start_time_utc,
                "timezone": profile.timezone,
            },
            "input": {
                "sample_rate_hz": batch.sample_rate_hz,
                "sample_count": len(batch.timestamp_s),
                "duration_seconds": batch.duration_s,
            },
            "summary": summary,
            "personal_baseline": baseline_score,
            "alerts": alerts,
            "frontend": frontend,
            "windows": [window.to_dict() for window in windows],
            "limitations": [
                "Mechanical pulse is not ECG and cannot diagnose arrhythmia.",
                "Sleep candidate is sustained immobility, not a validated sleep stage.",
                "Respiratory cessation can be sensor contact loss; airflow and oxygenation are not measured.",
                "Thresholds and signal settings require prospective validation on final VitalSense hardware.",
            ],
        }
    )


def _process_windows(batch: SensorBatch, config: AnalysisConfig) -> list[WindowResult]:
    starts = np.arange(
        batch.timestamp_s[0],
        batch.timestamp_s[-1] - config.minimum_window_seconds + 1e-9,
        config.step_seconds,
    )
    windows: list[WindowResult] = []
    for start_s in starts:
        stop_s = min(start_s + config.window_seconds, batch.timestamp_s[-1])
        start_index = int(np.searchsorted(batch.timestamp_s, start_s, side="left"))
        stop_index = int(np.searchsorted(batch.timestamp_s, stop_s, side="right"))
        if stop_index - start_index < 2:
            continue
        window_batch = batch.slice(start_index, stop_index)
        if window_batch.duration_s < config.minimum_window_seconds:
            continue
        windows.append(_process_window(window_batch, config))
    return windows


def _process_window(batch: SensorBatch, config: AnalysisConfig) -> WindowResult:
    features = characterize_window(batch, config)
    static_candidate = bool(features["static_candidate"])
    panting_candidate = bool(features["panting_candidate"])
    reason_codes = list(features["reason_codes"])
    piezo_impulse_artifact = bool(features["piezo_impulse_artifact_candidate"])
    usable_static = static_candidate and not panting_candidate and not piezo_impulse_artifact
    pulse = detect_pulses(batch, config) if usable_static else None
    respiration, imu_resp, piezo_resp = detect_respiration(batch, config) if usable_static else (None, None, None)

    piezo_quality = float(features["piezo_quality"])
    imu_quality = float(features["imu_quality"])
    detector_quality = 0.0
    if pulse is not None and respiration is not None:
        detector_quality = 0.55 * pulse.quality + 0.45 * respiration.quality
    fusion_quality = clamp(0.25 * piezo_quality + 0.25 * imu_quality + 0.50 * detector_quality)

    if pulse is None or pulse.rate_bpm is None:
        reason_codes.append("pulse_not_detected")
    if respiration is None or respiration.rate_bpm is None:
        reason_codes.append("respiration_not_detected")
    if imu_resp and piezo_resp and imu_resp.rate_bpm is not None and piezo_resp.rate_bpm is not None:
        if abs(imu_resp.rate_bpm - piezo_resp.rate_bpm) > config.respiration_agreement_bpm:
            reason_codes.append("respiration_modalities_disagree")

    valid = bool(
        usable_static
        and pulse is not None
        and pulse.rate_bpm is not None
        and respiration is not None
        and respiration.rate_bpm is not None
        and piezo_quality >= config.minimum_modality_quality
        and imu_quality >= config.minimum_modality_quality
        and fusion_quality >= config.minimum_fusion_quality
    )
    if not valid:
        reason_codes.append("quality_gate_failed")

    pulse_intervals = pulse.intervals_s if pulse else []
    breath_intervals = respiration.intervals_s if respiration else []
    heart_rate = pulse.rate_bpm if pulse else None
    respiratory_rate = respiration.rate_bpm if respiration else None
    coupling = cardiorespiratory_coupling(
        pulse.times_s if pulse else [],
        respiration.times_s if respiration else [],
        heart_rate,
        respiratory_rate,
    )
    signal_metrics = dict(features["signal_metrics"])
    dropout_runs: list[float] = []
    if usable_static:
        imu_resp_wave = safe_bandpass(
            principal_component(batch.acceleration), batch.sample_rate_hz, *config.respiration_band_hz
        )
        piezo_resp_wave = safe_bandpass(batch.piezo, batch.sample_rate_hz, *config.respiration_band_hz)
        dropout_runs = joint_amplitude_dropout_runs(
            imu_resp_wave, piezo_resp_wave, batch.sample_rate_hz, minimum_seconds=20.0
        )
        if dropout_runs:
            reason_codes.append("possible_resp_pause_or_signal_loss")
    signal_metrics["possible_resp_pause_count"] = float(len(dropout_runs))
    signal_metrics["longest_possible_resp_pause_s"] = max(dropout_runs) if dropout_runs else 0.0

    return WindowResult(
        start_s=float(batch.timestamp_s[0]),
        end_s=float(batch.timestamp_s[-1]),
        context="panting_or_unusable" if panting_candidate else ("quiet_awake_rest" if static_candidate else "moving"),
        orientation_bin=str(features["orientation_bin"]),
        static_candidate=static_candidate,
        panting_candidate=panting_candidate,
        piezo_quality=piezo_quality,
        imu_quality=imu_quality,
        fusion_quality=fusion_quality,
        valid=valid,
        reason_codes=sorted(set(reason_codes)),
        activity_enmo_mg=float(features["activity_enmo_mg"]),
        activity_intensity=float(features["activity_intensity"]),
        accel_dynamic_rms_g=float(features["accel_dynamic_rms_g"]),
        gyro_rms_dps=float(features["gyro_rms_dps"]),
        gravity_vector_g=list(features["gravity_vector_g"]),
        heart_rate_bpm=heart_rate,
        respiratory_rate_bpm=respiratory_rate,
        piezo_resp_rate_bpm=piezo_resp.rate_bpm if piezo_resp else None,
        imu_resp_rate_bpm=imu_resp.rate_bpm if imu_resp else None,
        pulse_count=len(pulse.times_s) if pulse else 0,
        breath_count=len(respiration.times_s) if respiration else 0,
        pulse_intervals_s=pulse_intervals,
        breath_intervals_s=breath_intervals,
        interbeat_metrics=interval_variability(pulse_intervals, prefix="ibi"),
        respiratory_variability=respiratory_variability(breath_intervals),
        cardiorespiratory_coupling=coupling,
        pulse_amplitude=pulse.amplitude if pulse else None,
        respiratory_amplitude=respiration.amplitude if respiration else None,
        signal_metrics=signal_metrics,
    )
