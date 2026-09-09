"""
Item 6: augmentation for the four worst-performing rare fine-grained
classes (Bowing, Jumping, Galloping, Tugging — all near-0% F1 in the
baseline fine model due to having 7-234 total samples across 45 dogs).

Augmented copies are generated from RAW window signal (jitter, time-warp,
rotation) and tagged with their source dog's DogID. To avoid leaking
synthetic data into evaluation, GroupKFold splitting happens on the REAL
windows only; augmented copies are added to a fold's training set only
when their source dog is itself in that fold's training set, and are never
added to any test set.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
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
RARE_CLASSES = ["Bowing", "Jumping", "Galloping", "Tugging"]
AUGMENTATIONS_PER_SAMPLE = 8  # jitter+warp+rotate combos per original rare-class window

rng = np.random.default_rng(RANDOM_STATE)


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


def jitter(raw6):
    sigma = 0.05 * (raw6.std(axis=0) + 1e-6)
    return raw6 + rng.normal(0, sigma, raw6.shape)


def time_warp(raw6, sigma=0.15):
    n = raw6.shape[0]
    n_knots = 4
    knot_x = np.linspace(0, n - 1, n_knots)
    knot_y = knot_x + rng.normal(0, sigma * n / n_knots, n_knots)
    knot_y[0], knot_y[-1] = 0, n - 1
    knot_y = np.sort(knot_y)
    warped_t = np.interp(np.arange(n), knot_x, knot_y)
    warped_t = np.clip(warped_t, 0, n - 1)
    out = np.empty_like(raw6)
    orig_t = np.arange(n)
    for c in range(raw6.shape[1]):
        out[:, c] = np.interp(warped_t, orig_t, raw6[:, c])
    return out


def rotate(raw6, max_angle_deg=15):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-max_angle_deg, max_angle_deg)
    r = Rotation.from_rotvec(np.radians(angle) * axis)
    accel = raw6[:, 0:3] @ r.as_matrix().T
    gyro = raw6[:, 3:6] @ r.as_matrix().T
    return np.concatenate([accel, gyro], axis=1)


def augment_window(raw6):
    out = raw6
    if rng.random() < 0.7:
        out = jitter(out)
    if rng.random() < 0.7:
        out = time_warp(out)
    if rng.random() < 0.7:
        out = rotate(out)
    return out


def build_augmented(windows, fine_labels, dog_id, neck_channels):
    """windows: (n, WINDOW, n_channels) with neck_channels order; returns
    augmented (windows, labels, dog_id) arrays for rare classes only."""
    raw_idx = [neck_channels.index(c) for c in
               ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]]
    mag_idx = {"ANeck_mag": neck_channels.index("ANeck_mag"),
               "GNeck_mag": neck_channels.index("GNeck_mag")}

    aug_windows, aug_labels, aug_dogs = [], [], []
    for cls in RARE_CLASSES:
        idxs = np.where(fine_labels == cls)[0]
        for i in idxs:
            raw6 = windows[i][:, raw_idx]
            for _ in range(AUGMENTATIONS_PER_SAMPLE):
                aug_raw6 = augment_window(raw6)
                full = np.empty_like(windows[i])
                full[:, raw_idx] = aug_raw6
                full[:, mag_idx["ANeck_mag"]] = np.sqrt((aug_raw6[:, 0:3] ** 2).sum(axis=1))
                full[:, mag_idx["GNeck_mag"]] = np.sqrt((aug_raw6[:, 3:6] ** 2).sum(axis=1))
                aug_windows.append(full)
                aug_labels.append(cls)
                aug_dogs.append(dog_id[i])
    print(f"[augment] generated {len(aug_windows):,} augmented windows from "
          f"{sum((fine_labels == c).sum() for c in RARE_CLASSES)} real rare-class windows")
    return np.stack(aug_windows), np.array(aug_labels), np.array(aug_dogs)


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    real_windows = data["windows"]
    real_feats = extract_window_features(real_windows)
    dog_id = data["dog_id"]
    y_fine = data["fine"]
    print(f"[window] {len(real_feats):,} real windows")

    aug_windows, aug_labels, aug_dogs = build_augmented(real_windows, y_fine, dog_id, neck_channels)
    aug_feats = extract_window_features(aug_windows)

    gkf = GroupKFold(n_splits=N_FOLDS)

    def run(use_augmentation):
        y_true_all, y_pred_all = [], []
        for fold, (tr, te) in enumerate(gkf.split(real_feats, y_fine, dog_id)):
            train_feats = real_feats[tr]
            train_labels = y_fine[tr]
            if use_augmentation:
                train_dogs = set(dog_id[tr])
                aug_mask = np.isin(aug_dogs, list(train_dogs))
                train_feats = np.concatenate([train_feats, aug_feats[aug_mask]], axis=0)
                train_labels = np.concatenate([train_labels, aug_labels[aug_mask]], axis=0)
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=18, class_weight="balanced_subsample",
                n_jobs=-1, random_state=RANDOM_STATE,
            )
            clf.fit(train_feats, train_labels)
            pred = clf.predict(real_feats[te])
            y_true_all.append(y_fine[te])
            y_pred_all.append(pred)
            print(f"  [{'augmented' if use_augmentation else 'baseline'}] fold {fold+1}/{N_FOLDS} "
                  f"train_size={len(train_feats):,}")
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        return report

    print("\n=== Baseline (no augmentation), for direct comparison on this exact run ===")
    baseline_report = run(use_augmentation=False)
    print("\n=== With augmentation (rare classes only) ===")
    augmented_report = run(use_augmentation=True)

    print("\n--- Rare-class F1 before vs after ---")
    comparison = {}
    for cls in RARE_CLASSES:
        before = baseline_report.get(cls, {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0})
        after = augmented_report.get(cls, {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0})
        comparison[cls] = {"before": before, "after": after}
        print(f"{cls}: F1 {before['f1-score']:.3f} -> {after['f1-score']:.3f} "
              f"(precision {before['precision']:.3f}->{after['precision']:.3f}, "
              f"recall {before['recall']:.3f}->{after['recall']:.3f}, support={before['support']})")

    print("\n--- Common-class metrics (unaffected classes, checking no dilution) ---")
    common_before_acc = baseline_report["accuracy"]
    common_after_acc = augmented_report["accuracy"]
    common_before_macro = baseline_report["macro avg"]["f1-score"]
    common_after_macro = augmented_report["macro avg"]["f1-score"]
    print(f"Overall accuracy: {common_before_acc:.4f} -> {common_after_acc:.4f}")
    print(f"Overall macro F1: {common_before_macro:.4f} -> {common_after_macro:.4f}")

    with open(MODELS_DIR / "augmentation_results.json", "w") as f:
        json.dump({
            "rare_class_comparison": comparison,
            "overall_accuracy": {"before": common_before_acc, "after": common_after_acc},
            "overall_macro_f1": {"before": common_before_macro, "after": common_after_macro},
            "full_baseline_report": baseline_report,
            "full_augmented_report": augmented_report,
        }, f, indent=2)
    print(f"\nSaved augmentation_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
