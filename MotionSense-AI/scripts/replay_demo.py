"""
Milestone 7: replay demo.

Feeds one dog's full recorded session through the actual shipped pipeline —
ONNX coarse model -> confidence gate -> ONNX active-fine model -> hierarchical
gating -> anomaly flags (gait regularity + personal-baseline IsolationForest)
— and plots predicted vs. true behavior over time, with anomaly flags
overlaid, to prove the exported pipeline works end-to-end on a real session.

Picks the dog with the most windows (richest, longest session) for the
demo, run through the FINAL model trained on all 45 dogs — this is a
pipeline/mechanics demo, not a fresh accuracy claim (those come from the
proper leave-one-dog-out CV already reported in PROJECT.md).
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import pandas as pd

from anomaly_detection import explain_flag, gait_regularity_scores
from features import (
    add_magnitude, build_windows, channel_list, extract_window_features,
    NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
EXPORT_DIR = ROOT / "export"
MODELS_DIR = ROOT / "models"

COARSE_CONF_THRESHOLD = 0.6
FINE_CONF_THRESHOLD = 0.5


def load_full_data():
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


def onnx_predict_proba(sess, X, class_names):
    """Runs the ONNX model and decodes its integer label output back to the
    original string class names. NOTE: the ONNX model itself only knows the
    integer-encoded labels it was trained on (via sklearn LabelEncoder in
    export_onnx.py) — it carries no class-name metadata, so the label
    encoder's class order must be shipped alongside the .onnx file for any
    real consumer (this was a real bug caught while building this demo: the
    first version used the raw ONNX integer output as if it were already
    the class name, which silently produced a ~0% "accuracy" number)."""
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: X.astype(np.float32)})
    pred_idx, proba_raw = out[0], out[1]
    if isinstance(proba_raw, list):
        # zipmap output: dict keys are the same integer-encoded labels
        prob_classes = sorted(proba_raw[0].keys())
        proba = np.array([[d[c] for c in prob_classes] for d in proba_raw])
    else:
        proba = proba_raw
    pred_names = np.asarray(class_names)[np.asarray(pred_idx, dtype=int)]
    return pred_names, proba


def main():
    df = load_full_data()
    dog_counts = df.groupby("DogID").size().sort_values(ascending=False)
    demo_dog = int(dog_counts.index[0])
    print(f"[select] using DogID={demo_dog} ({dog_counts.iloc[0]:,} labeled rows) for the replay demo")

    dog_df = df[df["DogID"] == demo_dog].copy()
    neck_channels = channel_list(use_back=False)
    data = build_windows(dog_df, neck_channels)
    windows = data["windows"]
    feats = extract_window_features(windows)
    t_center = data["t_center"]
    y_true_coarse = data["coarse"]
    y_true_fine = data["fine"]
    print(f"[window] {len(feats):,} windows for this dog's session(s)")

    coarse_export = joblib.load(EXPORT_DIR / "behavior_coarse_lgbm.joblib")
    fine_export = joblib.load(EXPORT_DIR / "behavior_active_fine_lgbm.joblib")
    coarse_class_names = coarse_export["label_encoder"].classes_
    fine_class_names = fine_export["label_encoder"].classes_

    coarse_sess = rt.InferenceSession(str(EXPORT_DIR / "behavior_coarse_lgbm.onnx"), providers=["CPUExecutionProvider"])
    fine_sess = rt.InferenceSession(str(EXPORT_DIR / "behavior_active_fine_lgbm.onnx"), providers=["CPUExecutionProvider"])

    coarse_pred, coarse_proba = onnx_predict_proba(coarse_sess, feats, coarse_class_names)
    coarse_conf = coarse_proba.max(axis=1)
    coarse_label = np.where(coarse_conf >= COARSE_CONF_THRESHOLD, coarse_pred, "uncertain")

    fine_pred, fine_proba = onnx_predict_proba(fine_sess, feats, fine_class_names)
    fine_conf = fine_proba.max(axis=1)

    effective_label = np.array(coarse_label, dtype=object)
    active_and_confident = (coarse_label == "Active")
    for i in np.where(active_and_confident)[0]:
        if fine_conf[i] >= FINE_CONF_THRESHOLD:
            effective_label[i] = fine_pred[i]
        else:
            effective_label[i] = "Active (unspecified)"

    # Anomaly flags using this dog's OWN stored personal baseline.
    anomaly_model = joblib.load(MODELS_DIR / "anomaly_baseline.joblib")
    dog_stats = anomaly_model["per_dog_baseline_stats"].get(demo_dog)
    regularity = gait_regularity_scores(windows, neck_channels)
    is_locomotion = np.isin(y_true_fine, ["Walking", "Trotting", "Pacing", "Galloping"])

    anomaly_flags = np.zeros(len(feats), dtype=bool)
    gait_flags = np.zeros(len(feats), dtype=bool)
    if dog_stats is not None:
        for i in range(len(feats)):
            exp = explain_flag(
                feats[i], dog_stats, anomaly_model["isolation_forest"],
                anomaly_model["feature_names"],
                gait_regularity=regularity[i] if is_locomotion[i] else None,
                gait_threshold=anomaly_model["gait_irregularity_threshold"] if is_locomotion[i] else None,
            )
            anomaly_flags[i] = exp["is_anomalous"]
            gait_flags[i] = bool(exp["gait"] and exp["gait"]["flagged"])

    # --- accuracy of this demo run against ground truth (sanity, not the official metric) ---
    coarse_match = np.mean((coarse_label == y_true_coarse) | (coarse_label == "uncertain"))
    print(f"[demo] coarse label matches true (or abstained as uncertain): {coarse_match:.3f}")
    print(f"[demo] {anomaly_flags.sum()} / {len(feats)} windows flagged anomalous (personal baseline)")
    print(f"[demo] {gait_flags.sum()} / {is_locomotion.sum()} locomotion windows flagged for gait irregularity")

    # --- plot ---
    coarse_order = ["Resting", "Other", "Active", "uncertain"]
    y_true_num = np.array([coarse_order.index(c) for c in y_true_coarse])
    y_pred_num = np.array([coarse_order.index(c) for c in coarse_label])

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, height_ratios=[3, 1])
    ax = axes[0]
    ax.step(t_center, y_true_num, where="post", label="True (coarse)", color="tab:blue", linewidth=1.5, alpha=0.8)
    ax.step(t_center, y_pred_num, where="post", label="Predicted (ONNX, gated)", color="tab:orange",
            linewidth=1.2, alpha=0.8, linestyle="--")
    ax.set_yticks(range(len(coarse_order)))
    ax.set_yticklabels(coarse_order)
    ax.set_ylabel("Behavior")
    ax.set_title(f"MotionSense replay demo — DogID {demo_dog}, {len(feats):,} windows, "
                 f"coarse match-or-uncertain rate {coarse_match:.1%}")
    ax.legend(loc="upper right")

    ax2 = axes[1]
    flag_any = anomaly_flags | gait_flags
    ax2.scatter(t_center[anomaly_flags], np.ones(anomaly_flags.sum()) * 1.0,
                marker="|", color="red", label="Personal-baseline anomaly")
    ax2.scatter(t_center[gait_flags], np.ones(gait_flags.sum()) * 0.5,
                marker="|", color="purple", label="Gait irregularity")
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Anomaly flags over the session")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = MODELS_DIR / f"replay_demo_dog{demo_dog}.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")

    with open(MODELS_DIR / "replay_demo_summary.json", "w") as f:
        json.dump({
            "dog_id": demo_dog,
            "n_windows": len(feats),
            "coarse_match_or_uncertain_rate": float(coarse_match),
            "n_anomaly_flags": int(anomaly_flags.sum()),
            "n_gait_flags": int(gait_flags.sum()),
            "n_locomotion_windows": int(is_locomotion.sum()),
            "plot_path": str(out_path.relative_to(ROOT)),
        }, f, indent=2)
    print(f"Saved replay_demo_summary.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
