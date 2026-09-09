from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PiezoConfig:
    """Frontend-friendly defaults for a 12-bit ADC (0..4095)."""

    adc_min: float = 0.0
    adc_max: float = 4095.0
    touch_delta: float = 30.0
    press_delta: float = 120.0
    abnormal_delta: float = 700.0
    clipping_margin: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)


def piezo_baseline(samples: np.ndarray, calibration_samples: int = 50) -> float:
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    return float(np.median(values[: min(calibration_samples, len(values))]))


def analyze_piezo_window(
    samples: np.ndarray,
    baseline: float | None = None,
    config: PiezoConfig | None = None,
) -> dict:
    cfg = config or PiezoConfig()
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "available": False,
            "state": "unavailable",
            "label": "No piezo data",
            "color": "#64748b",
            "alert": False,
        }

    base = piezo_baseline(values) if baseline is None else float(baseline)
    delta = np.abs(values - base)
    peak_index = int(np.argmax(delta))
    peak_delta = float(delta[peak_index])
    peak_raw = float(values[peak_index])
    clipped = bool(
        peak_raw <= cfg.adc_min + cfg.clipping_margin
        or peak_raw >= cfg.adc_max - cfg.clipping_margin
    ) and peak_delta >= cfg.press_delta
    spike_count = int(np.sum((delta[1:] >= cfg.press_delta) & (delta[:-1] < cfg.press_delta)))

    if peak_delta >= cfg.abnormal_delta or clipped:
        state, label, color, message, alert = (
            "abnormal_force",
            "Abnormal / too hard",
            "#dc2626",
            "Excessive piezo impact detected. Check the sensor and animal.",
            True,
        )
    elif peak_delta >= cfg.press_delta:
        state, label, color, message, alert = (
            "pressed", "Pressed", "#f59e0b", "Piezo press/spike detected.", False
        )
    elif peak_delta >= cfg.touch_delta:
        state, label, color, message, alert = (
            "touch", "Light touch", "#3b82f6", "Small piezo response detected.", False
        )
    else:
        state, label, color, message, alert = (
            "normal", "Normal", "#16a34a", "Piezo is near its baseline.", False
        )

    return {
        "available": True,
        "state": state,
        "label": label,
        "color": color,
        "message": message,
        "alert": alert,
        "raw_latest": float(values[-1]),
        "raw_peak": peak_raw,
        "baseline": base,
        "peak_delta": peak_delta,
        "rms_delta": float(np.sqrt(np.mean(delta**2))),
        "intensity_percent": float(min(100.0, 100.0 * peak_delta / cfg.abnormal_delta)),
        "spike_count": spike_count,
        "clipped": clipped,
        "thresholds": cfg.to_dict(),
        "note": "Piezo indicates changing force/impact, not reliable static pressure.",
    }
