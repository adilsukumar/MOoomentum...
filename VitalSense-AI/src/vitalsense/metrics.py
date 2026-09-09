"""Physiological and behavioral metric calculations."""

from __future__ import annotations

import math

import numpy as np

from .signal import clamp


def interval_variability(intervals_s: list[float], *, prefix: str = "ibi") -> dict[str, float | None]:
    values_ms = np.asarray(intervals_s, dtype=float) * 1000.0
    if len(values_ms) < 2:
        return {
            f"mean_{prefix}_ms": float(np.mean(values_ms)) if len(values_ms) else None,
            f"{prefix}_sd_ms": None,
            f"{prefix}_rmssd_ms": None,
            f"{prefix}_pnn50_pct": None,
            f"{prefix}_cv": None,
            f"{prefix}_poincare_sd1_ms": None,
            f"{prefix}_poincare_sd2_ms": None,
        }

    differences = np.diff(values_ms)
    standard_deviation = float(np.std(values_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.square(differences)))) if len(differences) else None
    pnn50 = float(np.mean(np.abs(differences) > 50.0) * 100.0) if len(differences) else None
    coefficient = float(standard_deviation / max(np.mean(values_ms), 1e-9))
    sd1 = float(rmssd / math.sqrt(2.0)) if rmssd is not None else None
    sd2_term = 2.0 * standard_deviation**2 - (rmssd**2) / 2.0 if rmssd is not None else -1.0
    sd2 = float(math.sqrt(max(sd2_term, 0.0))) if rmssd is not None else None
    return {
        f"mean_{prefix}_ms": float(np.mean(values_ms)),
        f"{prefix}_sd_ms": standard_deviation,
        f"{prefix}_rmssd_ms": rmssd,
        f"{prefix}_pnn50_pct": pnn50,
        f"{prefix}_cv": coefficient,
        f"{prefix}_poincare_sd1_ms": sd1,
        f"{prefix}_poincare_sd2_ms": sd2,
    }


def respiratory_variability(intervals_s: list[float]) -> dict[str, float | None]:
    values = np.asarray(intervals_s, dtype=float)
    if len(values) < 2:
        return {
            "mean_breath_interval_s": float(np.mean(values)) if len(values) else None,
            "breath_interval_sd_s": None,
            "breath_interval_rmssd_s": None,
            "breath_interval_cv": None,
            "irregular_breathing_index": None,
        }
    differences = np.diff(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.square(differences))))
    coefficient = float(standard_deviation / max(mean, 1e-9))
    return {
        "mean_breath_interval_s": mean,
        "breath_interval_sd_s": standard_deviation,
        "breath_interval_rmssd_s": rmssd,
        "breath_interval_cv": coefficient,
        "irregular_breathing_index": clamp(coefficient / 0.50),
    }


def cardiorespiratory_coupling(
    pulse_times_s: list[float],
    breath_times_s: list[float],
    heart_rate_bpm: float | None,
    respiratory_rate_bpm: float | None,
) -> dict[str, float | None]:
    ratio = None
    if heart_rate_bpm and respiratory_rate_bpm:
        ratio = float(heart_rate_bpm / respiratory_rate_bpm)

    if len(pulse_times_s) < 3 or len(breath_times_s) < 3:
        return {
            "beats_per_breath": ratio,
            "respiratory_phase_locking_value": None,
            "coupling_sample_count": 0.0,
        }

    pulses = np.asarray(pulse_times_s, dtype=float)
    breaths = np.asarray(breath_times_s, dtype=float)
    indices = np.searchsorted(breaths, pulses, side="right") - 1
    valid = (indices >= 0) & (indices < len(breaths) - 1)
    if not valid.any():
        return {
            "beats_per_breath": ratio,
            "respiratory_phase_locking_value": None,
            "coupling_sample_count": 0.0,
        }
    indices = indices[valid]
    valid_pulses = pulses[valid]
    cycle_lengths = breaths[indices + 1] - breaths[indices]
    good = cycle_lengths > 0
    phases = 2.0 * np.pi * (valid_pulses[good] - breaths[indices[good]]) / cycle_lengths[good]
    locking = float(np.abs(np.mean(np.exp(1j * phases)))) if len(phases) else None
    return {
        "beats_per_breath": ratio,
        "respiratory_phase_locking_value": locking,
        "coupling_sample_count": float(len(phases)),
    }


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"minimum": None, "p05": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }

