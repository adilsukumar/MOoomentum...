"""CSV and JSON input/output contracts."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .longitudinal import PersonalBaseline
from .models import AnalysisConfig, DogProfile, SensorBatch


SENSOR_COLUMNS = (
    "piezo",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
)


def load_sensor_csv(path: str | Path) -> SensorBatch:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Sensor CSV has no header")
        timestamp_column = "timestamp_s" if "timestamp_s" in reader.fieldnames else "timestamp_ns"
        required = {timestamp_column, "piezo", "accel_x_g", "accel_y_g", "accel_z_g"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Sensor CSV is missing required columns: {sorted(missing)}")
        data: dict[str, list[float]] = {"timestamp_s": []}
        data.update({name: [] for name in SENSOR_COLUMNS})
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row[timestamp_column])
                if timestamp_column == "timestamp_ns":
                    timestamp /= 1_000_000_000.0
                data["timestamp_s"].append(timestamp)
                for name in SENSOR_COLUMNS:
                    default = "0" if name.startswith("gyro_") else None
                    value = row.get(name, default)
                    if value in (None, ""):
                        raise ValueError(f"missing {name}")
                    data[name].append(float(value))
            except ValueError as error:
                raise ValueError(f"Invalid sensor value on CSV row {row_number}: {error}") from error
    return SensorBatch(**{name: np.asarray(values, dtype=float) for name, values in data.items()})


def save_sensor_csv(batch: SensorBatch, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("timestamp_s",) + SENSOR_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for values in zip(*(getattr(batch, name) for name in names), strict=True):
            writer.writerow([f"{float(value):.9g}" for value in values])


def load_profile(path: str | Path | None) -> DogProfile:
    if path is None:
        return DogProfile()
    payload = load_json(path)
    allowed = {item.name for item in fields(DogProfile)}
    return DogProfile(**{key: value for key, value in payload.items() if key in allowed})


def load_config(path: str | Path | None) -> AnalysisConfig:
    if path is None:
        return AnalysisConfig()
    payload = load_json(path)
    allowed = {item.name for item in fields(AnalysisConfig)}
    for band in ("heart_band_hz", "respiration_band_hz"):
        if band in payload:
            payload[band] = tuple(payload[band])
    return AnalysisConfig(**{key: value for key, value in payload.items() if key in allowed})


def load_baseline(path: str | Path | None) -> PersonalBaseline | None:
    return PersonalBaseline.from_dict(load_json(path)) if path else None


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

