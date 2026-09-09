"""Reusable signal-processing and quality helpers."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as scipy_signal


EPSILON = 1e-12


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))


def safe_bandpass(values: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = fill_nonfinite(values)
    nyquist = sample_rate_hz / 2.0
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if high <= low or len(values) < 16:
        return scipy_signal.detrend(values, type="linear")
    sos = scipy_signal.butter(4, [low, high], btype="bandpass", output="sos")
    return _safe_sos_filter(sos, scipy_signal.detrend(values, type="linear"))


def safe_highpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    values = fill_nonfinite(values)
    normalized = min(max(cutoff_hz / (sample_rate_hz / 2.0), 1e-5), 0.999)
    sos = scipy_signal.butter(3, normalized, btype="highpass", output="sos")
    return _safe_sos_filter(sos, values)


def _safe_sos_filter(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
    try:
        return scipy_signal.sosfiltfilt(sos, values)
    except ValueError:
        return scipy_signal.sosfilt(sos, values)


def fill_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(len(values))
    values[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return values


def spectral_concentration(
    values: np.ndarray,
    sample_rate_hz: float,
    target_band_hz: tuple[float, float],
    total_band_hz: tuple[float, float] = (0.05, 20.0),
) -> float:
    values = fill_nonfinite(values)
    if len(values) < 16 or robust_scale(values) <= EPSILON:
        return 0.0
    nperseg = min(len(values), max(64, int(sample_rate_hz * 20)))
    frequencies, power = scipy_signal.welch(values, sample_rate_hz, nperseg=nperseg)
    target = (frequencies >= target_band_hz[0]) & (frequencies <= target_band_hz[1])
    total = (frequencies >= total_band_hz[0]) & (frequencies <= min(total_band_hz[1], sample_rate_hz / 2.0))
    # np.trapezoid was introduced in NumPy 2.0; scipy still supports projects
    # running the minimum-compatible NumPy 1.x releases through np.trapz.
    integrate = getattr(np, "trapezoid", np.trapz)
    denominator = float(integrate(power[total], frequencies[total])) if total.any() else 0.0
    numerator = float(integrate(power[target], frequencies[target])) if target.any() else 0.0
    return clamp(numerator / max(denominator, EPSILON))


def dominant_frequency_hz(values: np.ndarray, sample_rate_hz: float, band_hz: tuple[float, float]) -> float | None:
    if len(values) < 16:
        return None
    frequencies, power = scipy_signal.welch(fill_nonfinite(values), sample_rate_hz, nperseg=min(len(values), int(sample_rate_hz * 30)))
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if not mask.any() or float(np.max(power[mask])) <= EPSILON:
        return None
    local = int(np.argmax(power[mask]))
    return float(frequencies[mask][local])


def clipping_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return 1.0
    low, high = np.min(finite), np.max(finite)
    if math.isclose(float(low), float(high)):
        return 1.0
    tolerance = max((high - low) * 1e-6, EPSILON)
    clipped = np.isclose(finite, low, atol=tolerance) | np.isclose(finite, high, atol=tolerance)
    return float(np.mean(clipped))


def timestamp_coverage(timestamp_s: np.ndarray) -> tuple[float, float]:
    differences = np.diff(timestamp_s)
    expected = float(np.median(differences))
    if expected <= 0:
        return 0.0, 1.0
    gap_fraction = float(np.mean(differences > expected * 1.5))
    jitter = robust_scale(differences) / expected
    return clamp(1.0 - gap_fraction - min(jitter, 1.0) * 0.25), gap_fraction


def principal_component(channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(channels, dtype=float)
    centered = channels - np.nanmedian(channels, axis=0)
    centered = np.nan_to_num(centered)
    if centered.ndim != 2 or centered.shape[1] == 1:
        return centered.ravel()
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[0]


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= EPSILON:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def orientation_bin(gravity_vector: np.ndarray) -> str:
    vector = np.asarray(gravity_vector, dtype=float)
    if np.linalg.norm(vector) <= EPSILON:
        return "unknown"
    axis = int(np.argmax(np.abs(vector)))
    sign = "+" if vector[axis] >= 0 else "-"
    return f"{sign}{'xyz'[axis]}"


def joint_amplitude_dropout_runs(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate_hz: float,
    minimum_seconds: float = 20.0,
    fraction_of_median: float = 0.15,
) -> list[float]:
    """Return long simultaneous amplitude dropouts; these are apnea-or-contact-loss candidates."""
    envelopes = []
    smoothing = max(1, int(sample_rate_hz))
    kernel = np.ones(smoothing, dtype=float) / smoothing
    for values in (first, second):
        envelope = np.abs(scipy_signal.hilbert(fill_nonfinite(values)))
        envelope = np.convolve(envelope, kernel, mode="same")
        reference = float(np.median(envelope))
        if reference <= EPSILON:
            return []
        envelopes.append(envelope / reference)
    low = (envelopes[0] < fraction_of_median) & (envelopes[1] < fraction_of_median)
    runs: list[float] = []
    start: int | None = None
    for index, is_low in enumerate(low):
        if is_low and start is None:
            start = index
        if not is_low and start is not None:
            duration = (index - start) / sample_rate_hz
            if duration >= minimum_seconds:
                runs.append(float(duration))
            start = None
    if start is not None:
        duration = (len(low) - start) / sample_rate_hz
        if duration >= minimum_seconds:
            runs.append(float(duration))
    return runs
