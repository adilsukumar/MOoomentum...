"""
Trains and evaluates the MotionSense behavior classifier on DogMoveData.csv.

Produces two models:
  - "neck"       : trained on neck-sensor features only (what a real collar
                    can actually provide)
  - "neck+back"   : trained on neck+back features (upper bound / reference,
                    NOT deployable without a harness)

Validation is leave-one-dog-out via GroupKFold grouped by DogID, so accuracy
reflects generalization to unseen dogs, not memorized gait quirks.

Usage:
    python train_behavior_model.py
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    feature_names, NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


def load_data(nrows=None):
    all_sensor_cols = NECK_AXES + BACK_AXES
    dtype = {c: "float32" for c in all_sensor_cols}
    dtype.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
    t0 = time.time()
    df = pd.read_csv(
        CSV_PATH,
        usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + all_sensor_cols,
        dtype=dtype,
        nrows=nrows,
    )
    print(f"[load] {len(df):,} rows in {time.time() - t0:.1f}s")

    df = df[df["Behavior_1"].isin(VALID_BEHAVIORS)].copy()
    df["Coarse"] = df["Behavior_1"].map(COARSE_MAP)
    print(f"[load] {len(df):,} labeled rows after dropping sync/undefined")

    add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
    add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")
    add_magnitude(df, ["ABack_x", "ABack_y", "ABack_z"], "ABack_mag")
    add_magnitude(df, ["GBack_x", "GBack_y", "GBack_z"], "GBack_mag")
    return df


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
        held_out_dogs = sorted(set(groups[te]))
        print(f"  [{label_name}] fold {fold+1}/{N_FOLDS} held-out dogs={held_out_dogs}")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    labels = sorted(set(y_true_all) | set(y_pred_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))
    return {"report": report, "confusion_matrix": cm.tolist(), "labels": labels}


def main():
    global N_FOLDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None, help="limit rows read (fast iteration/smoke test)")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="GroupKFold splits")
    args = parser.parse_args()
    N_FOLDS = args.folds

    df = load_data(nrows=args.nrows)

    print("[window] building windows + majority labels (this scans the full dataset)...")
    t0 = time.time()
    all_channels = channel_list(use_back=True)
    data = build_windows(df, all_channels)
    print(f"[window] {len(data['windows']):,} pure windows in {time.time() - t0:.1f}s")

    print("[feature] extracting per-window statistics...")
    feats_full = extract_window_features(data["windows"])
    names_full = feature_names(all_channels)

    neck_channels = channel_list(use_back=False)
    neck_idx = [all_channels.index(c) for c in neck_channels]
    # feats_full columns are grouped by stat then channel: [mean_c1..mean_cN, std_c1..]
    n_channels = len(all_channels)
    n_stats = feats_full.shape[1] // n_channels
    feats_full_3d = feats_full.reshape(len(feats_full), n_stats, n_channels)
    feats_neck = feats_full_3d[:, :, neck_idx].reshape(len(feats_full), -1)
    names_neck = feature_names(neck_channels)

    dog_id = data["dog_id"]
    y_fine = data["fine"]
    y_coarse = data["coarse"]

    results = {}
    for tag, X, names in [("neck", feats_neck, names_neck), ("neck+back", feats_full, names_full)]:
        print(f"\n=== {tag}: coarse (Active/Resting/Other) ===")
        results[f"{tag}_coarse"] = evaluate_cv(X, y_coarse, dog_id, f"{tag}-coarse")
        print(f"\n=== {tag}: fine-grained (17 behaviors) ===")
        results[f"{tag}_fine"] = evaluate_cv(X, y_fine, dog_id, f"{tag}-fine")

    # Final deployable model: neck-only, coarse labels, trained on ALL dogs.
    final_clf = RandomForestClassifier(
        n_estimators=200, max_depth=18, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    final_clf.fit(feats_neck, y_coarse)
    joblib.dump(
        {"model": final_clf, "feature_names": names_neck, "classes": list(final_clf.classes_)},
        MODELS_DIR / "behavior_neck_coarse.joblib",
    )

    final_fine_clf = RandomForestClassifier(
        n_estimators=200, max_depth=18, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    final_fine_clf.fit(feats_neck, y_fine)
    joblib.dump(
        {"model": final_fine_clf, "feature_names": names_neck, "classes": list(final_fine_clf.classes_)},
        MODELS_DIR / "behavior_neck_fine.joblib",
    )

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "n_windows": int(len(data["windows"])),
                "n_dogs": int(len(set(dog_id))),
                "window_seconds": 1.0,
                "results": {
                    k: {"labels": v["labels"], "confusion_matrix": v["confusion_matrix"],
                        "accuracy": v["report"].get("accuracy"),
                        "macro_f1": v["report"].get("macro avg", {}).get("f1-score"),
                        "per_class": {lbl: v["report"][lbl] for lbl in v["labels"] if lbl in v["report"]}}
                    for k, v in results.items()
                },
            },
            f, indent=2,
        )
    print(f"\nSaved models + metrics.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
