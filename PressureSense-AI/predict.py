from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.goat_motion.features import FeatureConfig, extract_window_features
from src.goat_motion.piezo import PiezoConfig, analyze_piezo_window, piezo_baseline


ALIASES = {
    "x": "acc_x", "y": "acc_y", "z": "acc_z",
    "ax": "acc_x", "ay": "acc_y", "az": "acc_z",
    "gx": "gyro_x", "gy": "gyro_y", "gz": "gyro_z",
    "piezo_value": "piezo", "piezo_raw": "piezo",
    "pos_x": "acc_x", "pos_y": "acc_y", "pos_z": "acc_z",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for column in frame.columns:
        key = column.strip().lower()
        rename[column] = ALIASES.get(key, key)
    return frame.rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run windowed goat behaviour inference")
    parser.add_argument("csv", help="BMI270/piezo CSV with acc_x, acc_y, acc_z columns")
    parser.add_argument("--model", default="models/goat_motion.joblib")
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None, help="Input Hz; resampled to model Hz")
    parser.add_argument("--acc-unit", choices=("g", "mg", "m/s2"), default="g")
    parser.add_argument("--piezo-press-threshold", type=float, default=120.0)
    parser.add_argument("--piezo-abnormal-threshold", type=float, default=700.0)
    parser.add_argument(
        "--profile", choices=("horn", "ear", "cabritrack", "mosar"), default="horn",
        help="Sensor mounting profile; horn supports all four target behaviours",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    raw_config = bundle["feature_config"]
    config = FeatureConfig(
        sample_rate_hz=float(raw_config["sample_rate_hz"]),
        window_seconds=float(raw_config["window_seconds"]),
        channels=tuple(raw_config["channels"]),
    )
    frame = normalize_columns(pd.read_csv(args.csv))
    piezo_values = (
        frame["piezo"].to_numpy(dtype=np.float32) if "piezo" in frame.columns else None
    )
    missing = [name for name in config.channels if name not in frame.columns]
    if missing:
        raise SystemExit(f"missing required columns: {', '.join(missing)}")
    values = frame[list(config.channels)].to_numpy(dtype=np.float32)
    acceleration_columns = [i for i, name in enumerate(config.channels) if name.startswith("acc_")]
    acceleration_scale = {"g": 1.0, "mg": 0.001, "m/s2": 1.0 / 9.80665}[args.acc_unit]
    values[:, acceleration_columns] *= acceleration_scale
    if args.sample_rate is not None and not np.isclose(args.sample_rate, config.sample_rate_hz):
        if args.sample_rate <= 0:
            raise SystemExit("--sample-rate must be positive")
        old_time = np.arange(len(values), dtype=np.float64) / args.sample_rate
        new_length = int(np.floor(old_time[-1] * config.sample_rate_hz)) + 1
        new_time = np.arange(new_length, dtype=np.float64) / config.sample_rate_hz
        values = np.column_stack([
            np.interp(new_time, old_time, values[:, column]) for column in range(values.shape[1])
        ]).astype(np.float32)
        if piezo_values is not None:
            piezo_values = np.interp(new_time, old_time, piezo_values).astype(np.float32)
    stride = round((args.stride_seconds or config.window_seconds) * config.sample_rate_hz)
    profile = {"horn": "cabritrack", "ear": "mosar"}.get(args.profile, args.profile)
    classifier = bundle.get("behavior_classifiers", {}).get(profile, bundle["behavior_classifier"])
    gait_detector = bundle.get("gait_detectors", {}).get(profile, bundle["gait_detector"])
    gait_threshold = bundle.get("gait_thresholds", {}).get(profile, bundle["gait_threshold"])
    walking_threshold = bundle.get("walking_thresholds", {}).get(profile, 0.5)
    piezo_config = PiezoConfig(
        press_delta=args.piezo_press_threshold,
        abnormal_delta=args.piezo_abnormal_threshold,
    )
    piezo_base = piezo_baseline(piezo_values) if piezo_values is not None else None
    output = []
    for start in range(0, len(values) - config.window_samples + 1, stride):
        features = extract_window_features(values[start : start + config.window_samples], config).reshape(1, -1)
        probabilities = classifier.predict_proba(features)[0]
        class_probabilities = {str(k): float(v) for k, v in zip(classifier.classes_, probabilities)}
        if class_probabilities.get("Walking", 0.0) >= walking_threshold:
            behavior = "Walking"
        else:
            behavior = max(
                (name for name in class_probabilities if name != "Walking"),
                key=class_probabilities.get,
            )
        gait_score = float(gait_detector.decision_function(features)[0])
        mobility = "not_assessed"
        if behavior == "Walking":
            mobility = "Normal gait" if gait_score >= gait_threshold else "Mobility anomaly"
        output.append({
            "start_seconds": start / config.sample_rate_hz,
            "end_seconds": (start + config.window_samples) / config.sample_rate_hz,
            "behavior": behavior,
            "confidence": float(np.max(probabilities)),
            "probabilities": class_probabilities,
            "mobility": mobility,
            "mobility_score": gait_score,
            "piezo": analyze_piezo_window(
                piezo_values[start : start + config.window_samples], piezo_base, piezo_config
            ) if piezo_values is not None else analyze_piezo_window(np.array([])),
        })
    if not output:
        raise SystemExit(
            f"input is shorter than one {config.window_seconds:g}-second model window "
            f"({config.window_samples} samples at {config.sample_rate_hz:g} Hz)"
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
