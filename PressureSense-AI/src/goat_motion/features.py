from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate_hz: float = 25.0
    window_seconds: float = 5.0
    channels: tuple[str, ...] = ("acc_x", "acc_y", "acc_z")

    @property
    def window_samples(self) -> int:
        return max(8, round(self.sample_rate_hz * self.window_seconds))

    def to_dict(self) -> dict:
        value = asdict(self)
        value["channels"] = list(self.channels)
        return value


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    if len(spectrum) <= 1 or np.max(spectrum[1:]) <= 1e-12:
        return 0.0
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    return float(frequencies[1 + np.argmax(spectrum[1:])])


def _frequency_features(values: np.ndarray, sample_rate_hz: float) -> list[float]:
    """Orientation-tolerant spectral shape and band-energy features."""
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    if len(power):
        power[0] = 0.0
    total = float(np.sum(power))
    if total <= 1e-12:
        return [0.0] * 6
    probability = power / total
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(max(2, len(power))))
    bands = ((0.2, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, sample_rate_hz / 2 + 1e-9))
    band_energy = [
        float(np.sum(power[(frequencies >= low) & (frequencies < high)]) / total)
        for low, high in bands
    ]
    centroid = float(np.sum(frequencies * power) / total)
    return [entropy, centroid, *band_energy]


def feature_names(channels: Iterable[str]) -> list[str]:
    per_channel = (
        "mean", "std", "min", "max", "median", "q25", "q75", "rms",
        "mad", "diff_std", "mean_abs_diff", "zero_cross_rate", "dominant_hz",
        "spectral_entropy", "spectral_centroid", "band_0p2_1", "band_1_3",
        "band_3_6", "band_6_nyquist",
    )
    names = [f"{channel}_{stat}" for channel in channels for stat in per_channel]
    names += [
        "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max",
        "acc_mag_rms", "acc_mag_diff_std", "acc_mag_dominant_hz",
        "acc_mag_spectral_entropy", "acc_mag_spectral_centroid",
        "acc_mag_band_0p2_1", "acc_mag_band_1_3", "acc_mag_band_3_6",
        "acc_mag_band_6_nyquist", "acc_mag_autocorr_1s", "acc_mag_autocorr_2s",
        "acc_xy_corr", "acc_xz_corr", "acc_yz_corr",
    ]
    return names


def extract_window_features(window: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Return orientation-tolerant time/frequency features for one sensor window.

    `window` must be shaped (samples, channels) in the same channel order as config.
    Missing optional BMI270 gyro/piezo channels should be supplied as zeros at inference.
    """
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(config.channels):
        raise ValueError(f"expected (samples, {len(config.channels)}) window, got {values.shape}")
    if values.shape[0] < 8 or not np.isfinite(values).all():
        raise ValueError("window needs at least 8 finite samples")

    result: list[float] = []
    for column in values.T:
        centered = column - np.mean(column)
        differences = np.diff(column)
        result.extend([
            float(np.mean(column)), float(np.std(column)), float(np.min(column)),
            float(np.max(column)), float(np.median(column)), float(np.quantile(column, 0.25)),
            float(np.quantile(column, 0.75)), float(np.sqrt(np.mean(column ** 2))),
            float(np.mean(np.abs(centered))), float(np.std(differences)),
            float(np.mean(np.abs(differences))), float(np.mean(centered[:-1] * centered[1:] < 0)),
            _dominant_frequency(column, config.sample_rate_hz),
        ])
        result.extend(_frequency_features(column, config.sample_rate_hz))

    channel_index = {name: index for index, name in enumerate(config.channels)}
    try:
        xyz = values[:, [channel_index["acc_x"], channel_index["acc_y"], channel_index["acc_z"]]]
    except KeyError as exc:
        raise ValueError("acc_x, acc_y and acc_z are required") from exc
    magnitude = np.linalg.norm(xyz, axis=1)
    result.extend([
        float(np.mean(magnitude)), float(np.std(magnitude)), float(np.min(magnitude)),
        float(np.max(magnitude)), float(np.sqrt(np.mean(magnitude ** 2))),
        float(np.std(np.diff(magnitude))), _dominant_frequency(magnitude, config.sample_rate_hz),
    ])
    result.extend(_frequency_features(magnitude, config.sample_rate_hz))
    centered_mag = magnitude - np.mean(magnitude)
    lag_1 = max(1, round(config.sample_rate_hz))
    lag_2 = max(1, round(2 * config.sample_rate_hz))
    result.extend([
        _safe_corr(centered_mag[:-lag_1], centered_mag[lag_1:]) if len(centered_mag) > lag_1 else 0.0,
        _safe_corr(centered_mag[:-lag_2], centered_mag[lag_2:]) if len(centered_mag) > lag_2 else 0.0,
        _safe_corr(xyz[:, 0], xyz[:, 1]), _safe_corr(xyz[:, 0], xyz[:, 2]),
        _safe_corr(xyz[:, 1], xyz[:, 2]),
    ])
    return np.asarray(result, dtype=np.float32)
