"""
Item 7: feature engineering pass. Adds signal magnitude area, jerk,
autocorrelation stride period, spectral entropy, and cross-axis correlation
(see features.extract_extended_features) on top of the existing feature
set, retrains the neck-only COARSE model (per the task's explicit scope),
and reports the accuracy delta against the already-known baseline number.
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
    extract_extended_features, feature_names,
    NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
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


def evaluate_cv(X, y, groups):
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
        print(f"  fold {fold+1}/{N_FOLDS} done")
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))
    return report, clf


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    base_feats = extract_window_features(data["windows"])
    base_names = feature_names(neck_channels)
    ext_feats, ext_names = extract_extended_features(data["windows"], neck_channels)
    print(f"[features] base={base_feats.shape[1]} extended_added={ext_feats.shape[1]} "
          f"total={base_feats.shape[1] + ext_feats.shape[1]}")

    combined = np.concatenate([base_feats, ext_feats], axis=1)
    dog_id = data["dog_id"]
    y_coarse = data["coarse"]

    print("\n=== Extended features: neck coarse ===")
    report, clf = evaluate_cv(combined, y_coarse, dog_id)

    # feature importance to see whether the new features actually pull weight
    importances = clf.feature_importances_
    all_names = base_names + ext_names
    top20 = sorted(zip(all_names, importances), key=lambda x: -x[1])[:20]
    new_feature_set = set(ext_names)
    n_new_in_top20 = sum(1 for name, _ in top20 if name in new_feature_set)
    print(f"\n[importance] {n_new_in_top20}/20 of the top-20 most important features are new (of {len(ext_names)} new features added)")
    for name, imp in top20:
        marker = " <- NEW" if name in new_feature_set else ""
        print(f"  {name}: {imp:.4f}{marker}")

    with open(MODELS_DIR / "extended_features_results.json", "w") as f:
        json.dump({
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "n_base_features": base_feats.shape[1],
            "n_extended_features": ext_feats.shape[1],
            "top20_importance": [{"feature": n, "importance": float(i), "is_new": n in new_feature_set} for n, i in top20],
        }, f, indent=2)
    print(f"\nSaved extended_features_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
