"""
Confirmation check: the neck-only coarse model's 96% accuracy came from a
RandomForest with a fixed random_state=42 and a deterministic GroupKFold
split (GroupKFold doesn't shuffle), so a literal re-run of
train_behavior_model.py would reproduce the exact same number — that
proves determinism, not robustness. This script retrains the same
neck-only coarse model across several different RandomForest random seeds
(same data, same GroupKFold folds, same features) to check the 96% holds
up and wasn't a lucky draw from one particular forest.
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
SEEDS = [1, 7, 123]


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


def evaluate_cv(X, y, groups, seed):
    gkf = GroupKFold(n_splits=N_FOLDS)
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=18, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(pred)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    return report["accuracy"], report["macro avg"]["f1-score"]


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    dog_id = data["dog_id"]
    y_coarse = data["coarse"]
    print(f"[window] {len(feats):,} windows")

    results = {}
    accs, f1s = [], []
    for seed in SEEDS:
        t0 = time.time()
        acc, macro_f1 = evaluate_cv(feats, y_coarse, dog_id, seed)
        accs.append(acc)
        f1s.append(macro_f1)
        results[seed] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"[seed={seed}] accuracy={acc:.4f} macro_f1={macro_f1:.4f} ({time.time()-t0:.1f}s)")

    print(f"\nOriginal run (seed=42): accuracy=0.9558 macro_f1=0.9497 (from models/metrics.json)")
    print(f"New seeds {SEEDS}: accuracy mean={np.mean(accs):.4f} std={np.std(accs):.4f} "
          f"(range {min(accs):.4f}-{max(accs):.4f})")
    print(f"New seeds macro_f1: mean={np.mean(f1s):.4f} std={np.std(f1s):.4f} "
          f"(range {min(f1s):.4f}-{max(f1s):.4f})")

    with open(MODELS_DIR / "seed_confirmation_results.json", "w") as f:
        json.dump({"original_seed_42": {"accuracy": 0.9557942349740457, "macro_f1": 0.9497156807463504},
                    "additional_seeds": results,
                    "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                    "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))}, f, indent=2)
    print(f"\nSaved seed_confirmation_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
