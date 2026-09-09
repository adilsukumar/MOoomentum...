"""
Unsupervised abnormal-movement detection for MotionSense.

IMPORTANT — read this before wiring output into any user-facing copy:
DogMoveData.csv contains zero disease/illness/injury labels. There is no
"limping" or "excessive scratching" ground truth anywhere in this dataset,
let alone a rabies or other-disease label. Nothing in this file is trained
against, or validated against, real abnormal/diseased dogs. It cannot
diagnose anything. What it legitimately does:

  1. Learns each dog's own normal-movement baseline from the neck-sensor
     feature space (using this dataset's healthy, labeled behavior data),
     and flags windows that are statistically unusual relative to that
     baseline (IsolationForest on per-dog-normalized features).
  2. Computes a gait-regularity score for locomotion windows (spectral
     purity of the dominant stride frequency) — irregular/asymmetric gait
     is a well-established general biomechanics signal that can accompany
     limping, but this script has never seen a single labeled limping
     example, so treat it as a heuristic, not a validated detector.
  3. Tracks each dog's baseline rate of classifier-predicted "Shaking"
     windows and flags sessions where that rate spikes well above the
     dog's own norm, as a rough proxy for elevated repetitive
     head/body-shake-like motion. Note "Shaking" in this dataset most
     likely means a body/head shake-off, NOT paw-directed scratching —
     this is a proxy for "more repetitive shake-like motion than usual,"
     not a scratching detector.

All outputs are "unusual pattern" flags with a plain-language reason, meant
to prompt "consider a vet check" — never a disease name. Disease names only
ever appear in the static educational content (see content/health_education.md),
never as model output.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    feature_names, NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"

LOCOMOTION_BEHAVIORS = {"Walking", "Trotting", "Pacing", "Galloping"}
GAIT_IRREGULARITY_PERCENTILE = 10    # flag the bottom 10% of regularity scores
SHAKING_SPIKE_MULTIPLIER = 3.0       # session rate vs personal baseline


def load_and_window(nrows=None):
    all_sensor_cols = NECK_AXES + BACK_AXES
    dtype = {c: "float32" for c in all_sensor_cols}
    dtype.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
    df = pd.read_csv(
        CSV_PATH,
        usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + all_sensor_cols,
        dtype=dtype, nrows=nrows,
    )
    df = df[df["Behavior_1"].isin(VALID_BEHAVIORS)].copy()
    df["Coarse"] = df["Behavior_1"].map(COARSE_MAP)
    add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
    add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")

    neck_channels = channel_list(use_back=False)
    data = build_windows(df, neck_channels)
    feats = extract_window_features(data["windows"])
    return data, feats, neck_channels


def gait_regularity_scores(windows, channels):
    """Spectral purity of the dominant frequency in ANeck_mag: higher = more
    of the window's energy sits in one clean periodic stride frequency,
    lower = power spread across many frequencies (irregular movement).

    A 1-second (100-sample) window only spans 1-3 dog stride cycles, so a
    plain rectangular-window FFT leaks a genuinely periodic signal's energy
    across neighboring bins regardless of how regular the gait actually is
    (confirmed empirically: an earlier version without a taper flagged 94%
    of ALL locomotion windows at a fixed 0.35 cutoff — that's a measurement
    artifact, not a fact about the dogs). A Hann taper reduces that leakage,
    and flagging is done by percentile within this run's own locomotion
    windows rather than a hand-picked absolute cutoff, since there is no
    externally-validated "this number means irregular gait" threshold."""
    mag_idx = channels.index("ANeck_mag")
    signal = windows[:, :, mag_idx]
    taper = np.hanning(signal.shape[1])
    spectrum = np.abs(np.fft.rfft(signal * taper, axis=1))
    spectrum[:, 0] = 0  # drop DC component
    total_power = spectrum.sum(axis=1) + 1e-9
    peak_power = spectrum.max(axis=1)
    return peak_power / total_power


def fit_personal_baseline(feats, dog_id):
    """Per-dog z-score normalization so the IsolationForest judges deviation
    from THIS dog's own typical movement, not raw cross-dog differences."""
    normed = np.zeros_like(feats)
    stats = {}
    for dog in np.unique(dog_id):
        mask = dog_id == dog
        mu = feats[mask].mean(axis=0)
        sigma = feats[mask].std(axis=0) + 1e-6
        normed[mask] = (feats[mask] - mu) / sigma
        stats[int(dog)] = {"mean": mu.tolist(), "std": sigma.tolist()}
    return normed, stats


def explain_flag(feat_vector, dog_stats, iso, feature_name_list,
                  gait_regularity=None, gait_threshold=None,
                  dog_shaking_rate=None, session_shaking_rate=None,
                  shaking_spike_multiplier=SHAKING_SPIKE_MULTIPLIER,
                  top_k=5):
    """Item 10: structured explanation for one window instead of a single
    opaque anomaly score. Returns which specific signal(s) drove a flag —
    IsolationForest itself gives no native per-feature attribution, so the
    proxy used here is: which individual features sit furthest (in z-score)
    from this dog's own baseline, since that per-dog-normalized vector is
    exactly what the forest scores. Combined with the two independent
    mechanical signals (gait regularity, shaking-rate deviation)."""
    mu = np.array(dog_stats["mean"])
    sigma = np.array(dog_stats["std"])
    z = (feat_vector - mu) / sigma
    anomaly_score = float(-iso.score_samples(z.reshape(1, -1))[0])
    is_anomalous = bool(iso.predict(z.reshape(1, -1))[0] == -1)

    top_idx = np.argsort(-np.abs(z))[:top_k]
    top_features = [{"feature": feature_name_list[i], "personal_zscore": float(z[i])} for i in top_idx]

    result = {
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous,
        "top_contributing_features": top_features,
        "gait": None,
        "shaking": None,
    }
    if gait_regularity is not None and gait_threshold is not None:
        result["gait"] = {
            "regularity_score": float(gait_regularity),
            "threshold": float(gait_threshold),
            "flagged": bool(gait_regularity < gait_threshold),
            "reason": "gait less regular than usual for this dog during locomotion" if gait_regularity < gait_threshold else None,
        }
    if dog_shaking_rate is not None and session_shaking_rate is not None:
        spike = session_shaking_rate > dog_shaking_rate * shaking_spike_multiplier and dog_shaking_rate > 0
        result["shaking"] = {
            "dog_baseline_rate": float(dog_shaking_rate),
            "observed_rate": float(session_shaking_rate),
            "flagged": bool(spike),
            "reason": "more shake-type motion than usual for this dog" if spike else None,
        }
    return result


def main():
    print("[load] windowing full dataset for anomaly baseline...")
    data, feats, channels = load_and_window()
    dog_id = data["dog_id"]
    fine = data["fine"]
    n = len(feats)
    print(f"[load] {n:,} windows across {len(set(dog_id))} dogs")

    normed_feats, per_dog_stats = fit_personal_baseline(feats, dog_id)

    iso = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
    )
    iso.fit(normed_feats)
    anomaly_score = -iso.score_samples(normed_feats)  # higher = more unusual

    regularity = gait_regularity_scores(data["windows"], channels)
    is_locomotion = np.isin(fine, list(LOCOMOTION_BEHAVIORS))
    gait_irregularity_threshold = float(
        np.percentile(regularity[is_locomotion], GAIT_IRREGULARITY_PERCENTILE)
    )
    gait_flag = is_locomotion & (regularity < gait_irregularity_threshold)

    shaking_rate_per_dog = {}
    for dog in np.unique(dog_id):
        mask = dog_id == dog
        shaking_rate_per_dog[int(dog)] = float((fine[mask] == "Shaking").mean())

    print("\n[summary] IsolationForest flagged "
          f"{(anomaly_score > np.percentile(anomaly_score, 95)).sum():,} / {n:,} "
          "windows as top-5% unusual (personal-baseline relative)")
    print(f"[summary] {gait_flag.sum():,} / {is_locomotion.sum():,} locomotion windows "
          f"(bottom {GAIT_IRREGULARITY_PERCENTILE}% by design) fall below the gait-regularity "
          f"threshold ({gait_irregularity_threshold:.4f}, set from this run's own data)")
    print("[summary] per-dog baseline Shaking-window rate (min/median/max): "
          f"{np.min(list(shaking_rate_per_dog.values())):.4f} / "
          f"{np.median(list(shaking_rate_per_dog.values())):.4f} / "
          f"{np.max(list(shaking_rate_per_dog.values())):.4f}")

    joblib.dump(
        {
            "isolation_forest": iso,
            "per_dog_baseline_stats": per_dog_stats,
            "shaking_rate_per_dog": shaking_rate_per_dog,
            "feature_names": feature_names(channels),
            "gait_irregularity_threshold": gait_irregularity_threshold,
            "gait_irregularity_percentile": GAIT_IRREGULARITY_PERCENTILE,
            "shaking_spike_multiplier": SHAKING_SPIKE_MULTIPLIER,
        },
        MODELS_DIR / "anomaly_baseline.joblib",
    )

    with open(MODELS_DIR / "anomaly_summary.json", "w") as f:
        json.dump(
            {
                "n_windows": int(n),
                "n_dogs": int(len(set(dog_id))),
                "isolation_forest_contamination": 0.05,
                "gait_irregularity_threshold": gait_irregularity_threshold,
                "gait_irregularity_percentile": GAIT_IRREGULARITY_PERCENTILE,
                "locomotion_windows_flagged": int(gait_flag.sum()),
                "locomotion_windows_total": int(is_locomotion.sum()),
                "shaking_rate_per_dog": shaking_rate_per_dog,
            },
            f, indent=2,
        )
    print(f"\nSaved anomaly_baseline.joblib + anomaly_summary.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
