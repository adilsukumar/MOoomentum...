"""
Item 4: LightGBM vs RandomForest head-to-head, same neck-only features,
same leave-one-dog-out GroupKFold split, coarse (3-class) and fine
(17-class) targets. LightGBM is the preferred export path since it
converts cleanly to ONNX (onnxmltools / onnxruntime), unlike sklearn's
RandomForest.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    feature_names, NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"
N_FOLDS = 5
RANDOM_STATE = 42


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


def evaluate_cv(X, y, groups, label_name, params):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    gkf = GroupKFold(n_splits=N_FOLDS)
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y_enc, groups)):
        t0 = time.time()
        clf = LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        clf.fit(X[tr], y_enc[tr])
        pred = clf.predict(X[te])
        y_true_all.append(y_enc[te])
        y_pred_all.append(pred)
        print(f"  [{label_name}] fold {fold+1}/{N_FOLDS} in {time.time()-t0:.1f}s")
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(
        le.inverse_transform(y_true_all), le.inverse_transform(y_pred_all),
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        le.inverse_transform(y_true_all), le.inverse_transform(y_pred_all), zero_division=0))
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")}}


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    dog_id = data["dog_id"]
    y_coarse = data["coarse"]
    y_fine = data["fine"]
    print(f"[window] {len(feats):,} windows")

    coarse_params = dict(n_estimators=300, num_leaves=31, class_weight="balanced")
    fine_params = dict(n_estimators=300, num_leaves=63, class_weight="balanced")

    print("\n=== LightGBM: neck coarse ===")
    coarse_result = evaluate_cv(feats, y_coarse, dog_id, "lgbm-coarse", coarse_params)
    print("\n=== LightGBM: neck fine ===")
    fine_result = evaluate_cv(feats, y_fine, dog_id, "lgbm-fine", fine_params)

    with open(MODELS_DIR / "lightgbm_results.json", "w") as f:
        import json
        json.dump({"neck_coarse": coarse_result, "neck_fine": fine_result}, f, indent=2)
    print(f"\nSaved lightgbm_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
