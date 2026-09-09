"""Mechanical pulse and respiratory-cycle detectors."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .models import AnalysisConfig, DetectedEvents, SensorBatch
from .signal import (
    clamp,
    dominant_frequency_hz,
    principal_component,
    robust_scale,
    safe_bandpass,
    spectral_concentration,
)


def detect_pulses(batch: SensorBatch, config: AnalysisConfig) -> DetectedEvents:
    sample_rate = batch.sample_rate_hz
    filtered = safe_bandpass(batch.piezo, sample_rate, *config.heart_band_hz)
    result = _detect_periodic_events(
        filtered,
        batch.timestamp_s,
        sample_rate,
        minimum_bpm=config.minimum_heart_bpm,
        maximum_bpm=config.maximum_heart_bpm,
        band_hz=config.heart_band_hz,
        source="piezo",
    )
    return result


def detect_respiration(batch: SensorBatch, config: AnalysisConfig) -> tuple[DetectedEvents, DetectedEvents, DetectedEvents]:
    sample_rate = batch.sample_rate_hz
    accel_component = principal_component(batch.acceleration)
    accel_filtered = safe_bandpass(accel_component, sample_rate, *config.respiration_band_hz)
    piezo_filtered = safe_bandpass(batch.piezo, sample_rate, *config.respiration_band_hz)

    imu = _detect_periodic_events(
        accel_filtered,
        batch.timestamp_s,
        sample_rate,
        minimum_bpm=config.minimum_resp_bpm,
        maximum_bpm=config.maximum_resp_bpm,
        band_hz=config.respiration_band_hz,
        source="bmi270",
    )
    piezo = _detect_periodic_events(
        piezo_filtered,
        batch.timestamp_s,
        sample_rate,
        minimum_bpm=config.minimum_resp_bpm,
        maximum_bpm=config.maximum_resp_bpm,
        band_hz=config.respiration_band_hz,
        source="piezo",
    )

    fused = _fuse_respiration(imu, piezo, config)
    return fused, imu, piezo


def _detect_periodic_events(
    filtered: np.ndarray,
    timestamp_s: np.ndarray,
    sample_rate_hz: float,
    minimum_bpm: float,
    maximum_bpm: float,
    band_hz: tuple[float, float],
    source: str,
) -> DetectedEvents:
    scale = robust_scale(filtered)
    if scale <= 1e-10:
        return DetectedEvents(source=source)

    frequency = _autocorrelation_frequency(filtered, sample_rate_hz, minimum_bpm, maximum_bpm)
    if frequency is None:
        frequency = dominant_frequency_hz(filtered, sample_rate_hz, band_hz)
    expected_rate = frequency * 60.0 if frequency else None
    physiologic_distance = sample_rate_hz * 60.0 / maximum_bpm * 0.90
    spectral_distance = sample_rate_hz / frequency * 0.60 if frequency else 0.0
    minimum_distance = max(1, int(max(physiologic_distance, spectral_distance)))
    candidates: list[DetectedEvents] = []

    for polarity in (1.0, -1.0):
        working = filtered * polarity
        peaks, properties = scipy_signal.find_peaks(
            working,
            distance=minimum_distance,
            prominence=max(scale * 0.45, np.std(working) * 0.20),
        )
        if len(peaks) < 2:
            continue
        times = timestamp_s[peaks]
        intervals = np.diff(times)
        valid_interval = (intervals >= 60.0 / maximum_bpm) & (intervals <= 60.0 / minimum_bpm)
        if not valid_interval.any():
            continue

        # Keep the longest physiologically plausible event sequence.
        best_start, best_stop = _longest_valid_run(valid_interval)
        selected_peaks = peaks[best_start : best_stop + 1]
        selected_times = timestamp_s[selected_peaks]
        selected_intervals = np.diff(selected_times)
        if len(selected_intervals) < 1:
            continue
        rate = float(60.0 / np.median(selected_intervals))
        interval_cv = float(np.std(selected_intervals) / max(np.mean(selected_intervals), 1e-9))
        band_quality = spectral_concentration(filtered, sample_rate_hz, band_hz)
        regularity = clamp(1.0 - interval_cv / 0.45)
        rate_agreement = 1.0
        if expected_rate:
            relative_error = abs(rate - expected_rate) / max(expected_rate, 1e-9)
            rate_agreement = clamp(1.0 - relative_error / 0.35)
        duration = max(timestamp_s[-1] - timestamp_s[0], 1.0)
        expected_events = duration * rate / 60.0
        coverage = clamp(len(selected_peaks) / max(expected_events, 1.0))
        quality = clamp(0.35 * band_quality + 0.30 * regularity + 0.20 * rate_agreement + 0.15 * coverage)
        amplitude = float(np.median(properties["prominences"])) if len(properties.get("prominences", [])) else None
        candidates.append(
            DetectedEvents(
                times_s=[float(value) for value in selected_times],
                intervals_s=[float(value) for value in selected_intervals],
                rate_bpm=rate,
                quality=quality,
                amplitude=amplitude,
                source=source,
            )
        )

    if not candidates:
        return DetectedEvents(source=source)
    return max(candidates, key=lambda item: (item.quality, len(item.times_s)))


def _autocorrelation_frequency(
    values: np.ndarray,
    sample_rate_hz: float,
    minimum_bpm: float,
    maximum_bpm: float,
) -> float | None:
    centered = values - np.mean(values)
    if np.std(centered) <= 1e-10:
        return None
    autocorrelation = scipy_signal.correlate(centered, centered, mode="full", method="fft")[len(centered) - 1 :]
    autocorrelation /= max(float(autocorrelation[0]), 1e-12)
    minimum_lag = max(1, int(sample_rate_hz * 60.0 / maximum_bpm))
    maximum_lag = min(len(autocorrelation) - 1, int(sample_rate_hz * 60.0 / minimum_bpm))
    if maximum_lag <= minimum_lag:
        return None
    segment = autocorrelation[minimum_lag : maximum_lag + 1]
    peaks, _ = scipy_signal.find_peaks(segment, prominence=0.02)
    if not len(peaks):
        lag = minimum_lag + int(np.argmax(segment))
    else:
        peak_values = segment[peaks]
        strongest = float(np.max(peak_values))
        # The strongest autocorrelation of a changing rate can occur at 2x/3x/4x
        # the beat period. Prefer the earliest well-supported peak (the fundamental).
        supported = peaks[peak_values >= max(strongest * 0.50, 0.08)]
        chosen = int(np.min(supported)) if len(supported) else int(peaks[int(np.argmax(peak_values))])
        lag = minimum_lag + chosen
    return float(sample_rate_hz / lag) if lag > 0 else None


def _longest_valid_run(valid_intervals: np.ndarray) -> tuple[int, int]:
    best_start = best_stop = 0
    current_start = 0
    for index, valid in enumerate(valid_intervals):
        if not valid:
            current_start = index + 1
            continue
        if index - current_start > best_stop - best_start:
            best_start = current_start
            best_stop = index + 1
    return best_start, best_stop


def _fuse_respiration(imu: DetectedEvents, piezo: DetectedEvents, config: AnalysisConfig) -> DetectedEvents:
    available = [result for result in (imu, piezo) if result.rate_bpm is not None]
    if not available:
        return DetectedEvents(source="none")
    if len(available) == 1:
        selected = available[0]
        return DetectedEvents(
            times_s=selected.times_s,
            intervals_s=selected.intervals_s,
            rate_bpm=selected.rate_bpm,
            quality=selected.quality * 0.80,
            amplitude=selected.amplitude,
            source=selected.source,
        )

    assert imu.rate_bpm is not None and piezo.rate_bpm is not None
    difference = abs(imu.rate_bpm - piezo.rate_bpm)
    agreement = clamp(1.0 - difference / max(config.respiration_agreement_bpm * 2.0, 1e-9))
    total_weight = max(imu.quality + piezo.quality, 1e-9)
    fused_rate = (imu.rate_bpm * imu.quality + piezo.rate_bpm * piezo.quality) / total_weight
    selected = max((imu, piezo), key=lambda item: item.quality)
    quality = clamp(0.55 * max(imu.quality, piezo.quality) + 0.25 * min(imu.quality, piezo.quality) + 0.20 * agreement)
    return DetectedEvents(
        times_s=selected.times_s,
        intervals_s=selected.intervals_s,
        rate_bpm=float(fused_rate),
        quality=quality,
        amplitude=selected.amplitude,
        source="fused" if difference <= config.respiration_agreement_bpm else f"selected_{selected.source}",
    )
