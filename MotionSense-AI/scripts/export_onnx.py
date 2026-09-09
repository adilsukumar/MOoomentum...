"""
Milestone 6: on-device export.

Trains the two production LightGBM models on the FULL dataset (all 45
dogs — CV was for validation, the shipped model uses all available data)
and converts each to ONNX, then verifies the ONNX Runtime output matches
the native LightGBM output on a held-out-from-training-order sample.

Two models, matching the hierarchical design adopted in the production
punch list:
  1. coarse: 3-class (Active/Resting/Other) neck-only classifier
  2. active_fine: 7-class (Active sub-behaviors only) neck-only classifier,
     used only when (1) predicts Active and confidence clears the
     item-3 gating threshold

The anomaly layer's IsolationForest is a separate, later export decision
(see the note at the bottom of this script's output) — not all sklearn
estimators convert to ONNX as cleanly as LightGBM does.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.preprocessing import LabelEncoder

from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    feature_names, NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "export"
EXPORT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
ACTIVE_FINE_CLASSES = [b for b, c in COARSE_MAP.items() if c == "Active"]


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


def train_and_export(X, y, name, params):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    clf.fit(X, y_enc)

    joblib.dump({"model": clf, "label_encoder": le}, EXPORT_DIR / f"{name}.joblib")

    onnx_model = convert_lightgbm(
        clf, initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
        target_opset=13,
    )
    onnx_path = EXPORT_DIR / f"{name}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Parity check: compare native vs ONNX Runtime on a random sample.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X[sample_idx].astype(np.float32)

    native_pred = clf.predict(X_sample)
    native_proba = clf.predict_proba(X_sample)

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_sample})
    onnx_pred = onnx_out[0]
    # onnxruntime's label output may be int64 encoded same as native_pred already
    onnx_proba_raw = onnx_out[1]  # list of dicts (per-class prob) in zipmap output, or array
    if isinstance(onnx_proba_raw, list):
        classes_sorted = sorted(onnx_proba_raw[0].keys())
        onnx_proba = np.array([[d[c] for c in classes_sorted] for d in onnx_proba_raw])
    else:
        onnx_proba = onnx_proba_raw

    label_match = float(np.mean(native_pred == onnx_pred))
    max_proba_diff = float(np.max(np.abs(native_proba - onnx_proba)))

    print(f"[{name}] classes={list(le.classes_)}")
    print(f"[{name}] label agreement (native vs ONNX): {label_match:.6f} "
          f"({int(label_match*len(sample_idx))}/{len(sample_idx)})")
    print(f"[{name}] max probability difference: {max_proba_diff:.8f}")

    return {
        "classes": list(le.classes_),
        "n_features": X.shape[1],
        "onnx_path": str(onnx_path.relative_to(ROOT)),
        "label_agreement": label_match,
        "max_proba_diff": max_proba_diff,
        "n_parity_samples": len(sample_idx),
    }


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    names = feature_names(neck_channels)
    y_coarse = data["coarse"]
    y_fine = data["fine"]
    print(f"[window] {len(feats):,} windows, {feats.shape[1]} features")

    results = {}

    print("\n=== Exporting coarse (3-class) model ===")
    results["coarse"] = train_and_export(
        feats, y_coarse, "behavior_coarse_lgbm",
        dict(n_estimators=300, num_leaves=31, class_weight="balanced"),
    )

    print("\n=== Exporting active-only fine (7-class) model ===")
    active_mask = y_coarse == "Active"
    results["active_fine"] = train_and_export(
        feats[active_mask], y_fine[active_mask], "behavior_active_fine_lgbm",
        dict(n_estimators=300, num_leaves=63, class_weight="balanced"),
    )

    with open(EXPORT_DIR / "feature_names.json", "w") as f:
        json.dump(names, f, indent=2)

    with open(EXPORT_DIR / "export_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ONNX models + parity results to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
