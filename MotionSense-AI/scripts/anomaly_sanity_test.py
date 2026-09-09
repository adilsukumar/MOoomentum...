"""
Item 9: synthetic anomaly-layer sanity test.

IMPORTANT — read the caveat before trusting these numbers: these
"abnormal" windows are synthetically corrupted normal Walking/Trotting
windows (reduced stride regularity, amplitude clipping, asymmetric
per-axis noise). They are engineering proxies for what limping-like or
restricted-movement signals MIGHT look like in IMU data — they are NOT
real pathology, were not reviewed by a vet, and passing this test proves
only that the anomaly layer reacts to gross synthetic distortion, not that
it would catch a real limping dog. Documented as a sanity check only.

Item 10: also demonstrates the explain_flag() structured output
(scripts/anomaly_detection.py) on both normal and corrupted examples, so
each flag can say what actually drove it — gait irregularity vs
personal-baseline deviation vs (session-level) shaking-rate spike — rather
than a single opaque score.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomaly_detection import explain_flag, gait_regularity_scores, load_and_window
from features import channel_list, extract_window_features, feature_names, NECK_AXES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

rng = np.random.default_rng(42)


def corrupt_reduce_stride_regularity(raw6, relative_sigma=0.5):
    """Break periodicity by adding a non-periodic random-walk drift, scaled
    to each channel's OWN std (a fixed absolute sigma is meaningless here:
    accel is O(0.01-2g), gyro is O(1-40 deg/s), so one fixed number is either
    negligible for gyro or huge for accel)."""
    channel_std = raw6.std(axis=0, keepdims=True) + 1e-6
    noise = np.cumsum(rng.normal(0, relative_sigma, raw6.shape) * channel_std, axis=0)
    noise -= noise.mean(axis=0)
    return raw6 + noise


def corrupt_clip_amplitude(raw6, clip_fraction=0.4):
    """Simulate restricted/limited range of motion (e.g. reduced stride
    height/reach) by compressing amplitude toward the per-channel mean."""
    mean = raw6.mean(axis=0, keepdims=True)
    return mean + (raw6 - mean) * clip_fraction


def corrupt_asymmetric_noise(raw6, axis=0, magnitude=0.6):
    """Simulate favoring one side (e.g. limping on one leg) by injecting a
    strong bias/noise burst on a single axis only."""
    out = raw6.copy()
    n = out.shape[0]
    burst = magnitude * np.sin(np.linspace(0, 6 * np.pi, n)) * (out[:, axis].std() + 1e-6)
    out[:, axis] += burst
    return out


# Two severity tiers per corruption: "mild" (first sanity-test run's original
# parameters) and "severe" (much larger perturbation) — run both to tell
# apart "the detector doesn't respond to subtle change" from "the detector
# doesn't respond at all, even to an obvious corruption."
CORRUPTIONS = {
    "reduced_stride_regularity_mild": lambda raw6: corrupt_reduce_stride_regularity(raw6, relative_sigma=0.15),
    "reduced_stride_regularity_severe": lambda raw6: corrupt_reduce_stride_regularity(raw6, relative_sigma=0.8),
    "amplitude_clipping_mild": lambda raw6: corrupt_clip_amplitude(raw6, clip_fraction=0.4),
    "amplitude_clipping_severe": lambda raw6: corrupt_clip_amplitude(raw6, clip_fraction=0.1),
    "asymmetric_axis_noise_mild": lambda raw6: corrupt_asymmetric_noise(raw6, magnitude=0.6),
    "asymmetric_axis_noise_severe": lambda raw6: corrupt_asymmetric_noise(raw6, magnitude=2.5),
}


def main():
    print("[load] loading real data + trained anomaly baseline...")
    anomaly_model = joblib.load(MODELS_DIR / "anomaly_baseline.joblib")
    data, feats, channels = load_and_window()
    fine = data["fine"]
    dog_id = data["dog_id"]

    locomotion_mask = np.isin(fine, ["Walking", "Trotting"])
    sample_idx = rng.choice(np.where(locomotion_mask)[0], size=200, replace=False)

    raw_idx = [channels.index(c) for c in ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]]
    mag_idx = {"ANeck_mag": channels.index("ANeck_mag"), "GNeck_mag": channels.index("GNeck_mag")}

    results = {}
    for corruption_name, corrupt_fn in CORRUPTIONS.items():
        original_windows = []
        corrupted_windows = []
        dogs = []
        for i in sample_idx:
            raw6 = data["windows"][i][:, raw_idx]
            corrupted6 = corrupt_fn(raw6)
            orig_full = data["windows"][i].copy()
            corrupt_full = data["windows"][i].copy()
            corrupt_full[:, raw_idx] = corrupted6
            corrupt_full[:, mag_idx["ANeck_mag"]] = np.sqrt((corrupted6[:, 0:3] ** 2).sum(axis=1))
            corrupt_full[:, mag_idx["GNeck_mag"]] = np.sqrt((corrupted6[:, 3:6] ** 2).sum(axis=1))
            original_windows.append(orig_full)
            corrupted_windows.append(corrupt_full)
            dogs.append(dog_id[i])
        original_windows = np.stack(original_windows)
        corrupted_windows = np.stack(corrupted_windows)
        dogs = np.array(dogs)

        orig_feats = extract_window_features(original_windows)
        corrupt_feats = extract_window_features(corrupted_windows)
        orig_regularity = gait_regularity_scores(original_windows, channels)
        corrupt_regularity = gait_regularity_scores(corrupted_windows, channels)

        orig_flagged, corrupt_flagged = 0, 0
        example_explanations = []
        for j in range(len(sample_idx)):
            dog = int(dogs[j])
            dog_stats = anomaly_model["per_dog_baseline_stats"].get(str(dog)) or anomaly_model["per_dog_baseline_stats"].get(dog)
            if dog_stats is None:
                continue
            orig_exp = explain_flag(
                orig_feats[j], dog_stats, anomaly_model["isolation_forest"],
                anomaly_model["feature_names"], gait_regularity=orig_regularity[j],
                gait_threshold=anomaly_model["gait_irregularity_threshold"],
            )
            corrupt_exp = explain_flag(
                corrupt_feats[j], dog_stats, anomaly_model["isolation_forest"],
                anomaly_model["feature_names"], gait_regularity=corrupt_regularity[j],
                gait_threshold=anomaly_model["gait_irregularity_threshold"],
            )
            orig_flagged += orig_exp["is_anomalous"] or (orig_exp["gait"] and orig_exp["gait"]["flagged"])
            corrupt_flagged += corrupt_exp["is_anomalous"] or (corrupt_exp["gait"] and corrupt_exp["gait"]["flagged"])
            if j < 2:
                example_explanations.append({"corruption": corruption_name, "original": orig_exp, "corrupted": corrupt_exp})

        results[corruption_name] = {
            "n_samples": len(sample_idx),
            "original_flagged_fraction": orig_flagged / len(sample_idx),
            "corrupted_flagged_fraction": corrupt_flagged / len(sample_idx),
            "example_explanations": example_explanations,
        }
        print(f"[{corruption_name}] original flagged: {orig_flagged}/{len(sample_idx)} "
              f"({orig_flagged/len(sample_idx):.1%}) | corrupted flagged: {corrupt_flagged}/{len(sample_idx)} "
              f"({corrupt_flagged/len(sample_idx):.1%})")

    with open(MODELS_DIR / "anomaly_sanity_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved anomaly_sanity_test_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
