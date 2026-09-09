from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Iterator

import numpy as np
import pandas as pd

from .features import FeatureConfig, extract_window_features


CABRITRACK_LABELS = {
    "Displacement": "Walking",
    "Grazing": "Grazing",
    "Ruminating_Chewing": "Rumination",
    "Resting": "Resting",
}
ZENODO_LABELS = {"Rumiando": "Rumination", "Inactiva": "Resting"}
ZENODO_TARGET_LABELS = {
    "Comiendo": "Grazing",
    "Inactiva": "Resting",
    "Tumbada": "Resting",
    "Rumiando": "Rumination",
    "Caminando": "Walking",
    "Desplazandose": "Walking",
    "Corriendo": "Walking",
}


def _windows(values: np.ndarray, length: int) -> Iterator[np.ndarray]:
    for start in range(0, len(values) - length + 1, length):
        yield values[start : start + length]


def _resample(values: np.ndarray, output_length: int) -> np.ndarray:
    if len(values) == output_length:
        return values
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, output_length)
    return np.column_stack([np.interp(new_x, old_x, values[:, i]) for i in range(values.shape[1])])


def load_cabritrack_features(
    path: str | Path,
    config: FeatureConfig,
    max_windows_per_class: int = 12_000,
    chunksize: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream CabriTrack and create non-overlapping, single-sequence windows."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    counts: Counter[str] = Counter()
    carry = pd.DataFrame()
    columns = ["Animal_id", "Behaviour", "X", "Y", "Z", "Sequence_num"]

    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, on_bad_lines="skip"):
        chunk = pd.concat([carry, chunk], ignore_index=True)
        if chunk.empty:
            continue
        last_key = tuple(chunk.iloc[-1][["Animal_id", "Sequence_num"]])
        is_last = (chunk["Animal_id"] == last_key[0]) & (chunk["Sequence_num"] == last_key[1])
        carry = chunk.loc[is_last].copy()
        ready = chunk.loc[~is_last]
        for (animal, _sequence), frame in ready.groupby(["Animal_id", "Sequence_num"], sort=False):
            source_label = str(frame["Behaviour"].iloc[0])
            label = CABRITRACK_LABELS.get(source_label)
            if label is None or counts[label] >= max_windows_per_class:
                continue
            xyz = frame[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
            for window in _windows(xyz, config.window_samples):
                if counts[label] >= max_windows_per_class:
                    break
                features.append(extract_window_features(window, config))
                labels.append(label)
                groups.append(f"cabritrack:{animal}")
                counts[label] += 1

    if not features:
        raise ValueError(f"no usable labeled windows found in {path}")
    return np.vstack(features), np.asarray(labels), np.asarray(groups)


def load_mosar_features(
    directory: str | Path,
    config: FeatureConfig,
    source_sample_rate_hz: float = 5.0,
    max_windows_per_class: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one or more MoSAR ear-tag CSVs and resample their labeled runs.

    Feeder activity is intentionally not called grazing: these indoor recordings
    do not distinguish pasture grazing. Clean standing/lying periods become
    Resting, while explicit ruminating and walking labels take precedence.
    """
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    counts: Counter[str] = Counter()
    source_window = max(8, round(config.window_seconds * source_sample_rate_hz))
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise ValueError(f"no CSV files found in {directory}")

    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "ACCx", "ACCy", "ACCz", "feeding_behav_data_goat",
            "position_behav_data_goat", "social_behav_data_goat", "other_behav_data_goat",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        target = pd.Series(pd.NA, index=frame.index, dtype="string")
        walking = frame["position_behav_data_goat"].eq("walking")
        ruminating = frame["feeding_behav_data_goat"].eq("ruminating") & ~walking
        quiet_posture = frame["position_behav_data_goat"].isin(["lying", "lyingd", "standing", "standingp"])
        quiet_context = (
            frame["feeding_behav_data_goat"].eq("nonef")
            & frame["social_behav_data_goat"].eq("nones")
            & frame["other_behav_data_goat"].eq("noneo")
        )
        target.loc[quiet_posture & quiet_context] = "Resting"
        target.loc[ruminating] = "Rumination"
        target.loc[walking] = "Walking"
        valid_acc = frame[["ACCx", "ACCy", "ACCz"]].notna().all(axis=1)
        usable = frame.loc[target.notna() & valid_acc, ["ACCx", "ACCy", "ACCz"]].copy()
        usable["target"] = target.loc[usable.index]
        # Missing accelerometer rows are expected at 5 Hz annotation alignment;
        # a new run starts only on label change or a gap larger than one second.
        gaps = usable.index.to_series().diff().fillna(1).gt(6)
        changes = usable["target"].ne(usable["target"].shift()).fillna(True)
        usable["run"] = (gaps | changes).cumsum()
        animal = path.stem
        for _run, run in usable.groupby("run", sort=False):
            label = str(run["target"].iloc[0])
            if counts[label] >= max_windows_per_class:
                continue
            xyz = run[["ACCx", "ACCy", "ACCz"]].to_numpy(dtype=np.float32)
            for window in _windows(xyz, source_window):
                if counts[label] >= max_windows_per_class:
                    break
                resampled = _resample(window, config.window_samples)
                features.append(extract_window_features(resampled, config))
                labels.append(label)
                groups.append(f"mosar:{animal}")
                counts[label] += 1

    if not features:
        raise ValueError(f"no usable MoSAR windows found in {directory}")
    return np.vstack(features), np.asarray(labels), np.asarray(groups)


def load_zenodo_features(
    directory: str | Path,
    config: FeatureConfig,
    max_windows_per_class: int = 12_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the UPV neck-mounted LIS3DH recordings with per-file rate inference."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    counts: Counter[str] = Counter()
    paths = sorted(Path(directory).glob("20*_LIS3DH_*.csv"))
    if not paths:
        raise ValueError(f"no Zenodo LIS3DH CSV files found in {directory}")
    for path in paths:
        match = re.search(r"_AW\d+_(\d+)_GUPV", path.name)
        if not match:
            continue
        animal = match.group(1)
        frame = pd.read_csv(path, usecols=["pos_x", "pos_y", "pos_z", "label", "time_stamp"])
        frame["target"] = frame["label"].map(ZENODO_TARGET_LABELS)
        frame["time"] = pd.to_datetime(frame["time_stamp"], errors="coerce")
        frame = frame.dropna(subset=["pos_x", "pos_y", "pos_z", "target", "time"]).copy()
        if len(frame) < 8:
            continue
        positive_deltas = frame["time"].diff().dt.total_seconds()
        median_delta = float(positive_deltas[positive_deltas > 0].median())
        if not np.isfinite(median_delta) or median_delta <= 0:
            continue
        source_rate = 1.0 / median_delta
        source_window = max(8, round(config.window_seconds * source_rate))
        gaps = positive_deltas.gt(max(0.5, 3.0 * median_delta)).fillna(False)
        changes = frame["target"].ne(frame["target"].shift()).fillna(True)
        frame["run"] = (gaps | changes).cumsum()
        for _run, run in frame.groupby("run", sort=False):
            label = str(run["target"].iloc[0])
            if counts[label] >= max_windows_per_class:
                continue
            # UPV values are m/s^2; the CabriTrack model contract uses g.
            xyz = run[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32) / 9.80665
            for window in _windows(xyz, source_window):
                if counts[label] >= max_windows_per_class:
                    break
                features.append(extract_window_features(_resample(window, config.window_samples), config))
                labels.append(label)
                groups.append(f"zenodo:{animal}")
                counts[label] += 1
    if not features:
        raise ValueError(f"no usable labeled Zenodo windows found in {directory}")
    return np.vstack(features), np.asarray(labels), np.asarray(groups)
