"""
Item 5: per-dog baseline (calibration) normalization.

Simulates an onboarding calibration: for each dog, use the first 60-90s of
its EARLIEST test session as a calibration segment, compute that segment's
per-feature mean/std, and z-score ALL of that dog's windows (every session)
against its own calibration baseline before classification. This is meant
to strip out dog-to-dog offset/scale differences (collar fit, size, resting
posture quirks) that the raw features otherwise bake in — exactly the kind
of variation that could hurt leave-one-dog-out generalization.

Compares directly against the uncalibrated neck-coarse/neck-fine numbers
already in models/metrics.json (same split, same features, only the
normalization step differs) rather than re-deriving a baseline.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold

from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"
N_FOLDS = 5
RANDOM_STATE = 42
CALIBRATION_SECONDS = 75  # midpoint of the requested 60-90s range
MIN_CALIBRATION_WINDOWS = 10  # fall back to no normalization below this


def load_data():
    all_sensor_cols = NECK_AXES + BACK_AXES
    dtype = {c: "float32" for c in all_sensor_cols}
    dtype.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
    df = pd.read_csv(
        CSV_PATH,
        usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + all_sensor_cols,
        dtype=dtype,
    )
    df = df[df["Behavior_1"].isin(VALID_BEHAVIORS)].copy()
    df["Coarse"] = df["Behavior_1"].map(COARSE_MAP)
    add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
    add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")
    return df


def calibrate(feats, dog_id, test_num, t_center):
    """Z-score every dog's features against that dog's own first
    CALIBRATION_SECONDS of its earliest session."""
    normed = np.array(feats, copy=True)
    n_fallback = 0
    for dog in np.unique(dog_id):
        mask = dog_id == dog
        first_test = test_num[mask].min()
        calib_mask = mask & (test_num == first_test) & (t_center <= t_center[mask & (test_num == first_test)].min() + CALIBRATION_SECONDS)
        if calib_mask.sum() < MIN_CALIBRATION_WINDOWS:
            n_fallback += 1
            mu = feats[mask].mean(axis=0)
            sigma = feats[mask].std(axis=0) + 1e-6
        else:
            mu = feats[calib_mask].mean(axis=0)
            sigma = feats[calib_mask].std(axis=0) + 1e-6
        normed[mask] = (feats[mask] - mu) / sigma
    print(f"[calibrate] {n_fallback} / {len(np.unique(dog_id))} dogs fell back to "
          f"whole-dog normalization (insufficient calibration-window data)")
    return normed


def evaluate_cv(X, y, groups, label_name):
    gkf = GroupKFold(n_splits=N_FOLDS)
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=18, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(pred)
        print(f"  [{label_name}] fold {fold+1}/{N_FOLDS} done")
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    dog_id, test_num, t_center = data["dog_id"], data["test_num"], data["t_center"]
    y_coarse, y_fine = data["coarse"], data["fine"]
    print(f"[window] {len(feats):,} windows")

    print("[calibrate] computing per-dog calibration baselines...")
    calibrated_feats = calibrate(feats, dog_id, test_num, t_center)

    print("\n=== Calibrated: neck coarse ===")
    coarse_result = evaluate_cv(calibrated_feats, y_coarse, dog_id, "calibrated-coarse")
    print("\n=== Calibrated: neck fine ===")
    fine_result = evaluate_cv(calibrated_feats, y_fine, dog_id, "calibrated-fine")

    with open(MODELS_DIR / "calibrated_results.json", "w") as f:
        json.dump({"neck_coarse": coarse_result, "neck_fine": fine_result,
                    "calibration_seconds": CALIBRATION_SECONDS}, f, indent=2)
    print(f"\nSaved calibrated_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
