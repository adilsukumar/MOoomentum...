"""
Exports out-of-fold (OOF) predictions + probabilities + per-window metadata
for the neck-only coarse and fine models, under proper leave-one-dog-out
GroupKFold (every window is predicted by a model that never saw its dog).

This one export feeds three separate experiments (temporal smoothing,
confidence thresholding, hierarchical gating) so they don't each require
their own multi-fold retrain.

Also trains a dedicated "Active-only" fine classifier (7 classes: the
behaviors that are actually Active) for the hierarchical-gating experiment,
since restricting the full 17-class model post-hoc to Active predictions is
a weaker architecture than a classifier that only ever had to distinguish
Active sub-behaviors from each other.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
ACTIVE_FINE_CLASSES = [b for b, c in COARSE_MAP.items() if c == "Active"]


def load_data():
    all_sensor_cols = NECK_AXES + BACK_AXES
    dtype = {c: "float32" for c in all_sensor_cols}
    dtype.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
    t0 = time.time()
    df = pd.read_csv(
        CSV_PATH,
        usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + all_sensor_cols,
        dtype=dtype,
    )
    print(f"[load] {len(df):,} rows in {time.time() - t0:.1f}s")
    df = df[df["Behavior_1"].isin(VALID_BEHAVIORS)].copy()
    df["Coarse"] = df["Behavior_1"].map(COARSE_MAP)
    add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
    add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")
    return df


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    t0 = time.time()
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    n = len(feats)
    print(f"[window] {n:,} windows in {time.time() - t0:.1f}s")

    dog_id = data["dog_id"]
    test_num = data["test_num"]
    t_center = data["t_center"]
    y_coarse = data["coarse"]
    y_fine = data["fine"]

    pred_coarse = np.empty(n, dtype=object)
    proba_coarse_max = np.zeros(n, dtype=np.float32)
    pred_fine = np.empty(n, dtype=object)
    proba_fine_max = np.zeros(n, dtype=np.float32)

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, te) in enumerate(gkf.split(feats, y_coarse, dog_id)):
        t0 = time.time()
        clf_c = RandomForestClassifier(
            n_estimators=200, max_depth=18, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        clf_c.fit(feats[tr], y_coarse[tr])
        proba = clf_c.predict_proba(feats[te])
        idx = proba.argmax(axis=1)
        pred_coarse[te] = clf_c.classes_[idx]
        proba_coarse_max[te] = proba[np.arange(len(te)), idx]

        clf_f = RandomForestClassifier(
            n_estimators=200, max_depth=18, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        clf_f.fit(feats[tr], y_fine[tr])
        probaf = clf_f.predict_proba(feats[te])
        idxf = probaf.argmax(axis=1)
        pred_fine[te] = clf_f.classes_[idxf]
        proba_fine_max[te] = probaf[np.arange(len(te)), idxf]
        print(f"[fold {fold+1}/{N_FOLDS}] done in {time.time()-t0:.1f}s "
              f"(held out {len(set(dog_id[te]))} dogs)")

    # Dedicated Active-only fine classifier (hierarchical gating, item 3)
    active_mask = y_coarse == "Active"
    feats_active = feats[active_mask]
    y_fine_active = y_fine[active_mask]
    dog_active = dog_id[active_mask]

    pred_fine_active = np.empty(len(feats_active), dtype=object)
    proba_fine_active_max = np.zeros(len(feats_active), dtype=np.float32)
    for fold, (tr, te) in enumerate(gkf.split(feats_active, y_fine_active, dog_active)):
        clf_a = RandomForestClassifier(
            n_estimators=200, max_depth=18, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        clf_a.fit(feats_active[tr], y_fine_active[tr])
        probaa = clf_a.predict_proba(feats_active[te])
        idxa = probaa.argmax(axis=1)
        pred_fine_active[te] = clf_a.classes_[idxa]
        proba_fine_active_max[te] = probaa[np.arange(len(te)), idxa]
        print(f"[active-fine fold {fold+1}/{N_FOLDS}] done")

    out = {
        "dog_id": dog_id, "test_num": test_num, "t_center": t_center,
        "y_coarse": y_coarse, "pred_coarse": pred_coarse, "proba_coarse_max": proba_coarse_max,
        "y_fine": y_fine, "pred_fine": pred_fine, "proba_fine_max": proba_fine_max,
        "active_mask": active_mask,
        "pred_fine_active": pred_fine_active, "proba_fine_active_max": proba_fine_active_max,
        "y_fine_active": y_fine_active,
    }
    joblib.dump(out, MODELS_DIR / "oof_predictions.joblib")
    print(f"\nSaved OOF predictions to {MODELS_DIR / 'oof_predictions.joblib'}")


if __name__ == "__main__":
    main()
