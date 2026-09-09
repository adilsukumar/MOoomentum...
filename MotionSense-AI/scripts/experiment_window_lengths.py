"""Compare longer temporal windows under the same dog-held-out protocol."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import features as F

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"


def load_data():
    cols = F.NECK_AXES + F.BACK_AXES
    dtype = {c: "float32" for c in cols}
    dtype.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
    df = pd.read_csv(CSV_PATH, usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + cols, dtype=dtype)
    df = df[df["Behavior_1"].isin(F.VALID_BEHAVIORS)].copy()
    df["Coarse"] = df["Behavior_1"].map(F.COARSE_MAP)
    F.add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
    F.add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")
    return df


def evaluate(X, y, groups, n_splits=5):
    le = LabelEncoder()
    ye = le.fit_transform(y)
    true, pred = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits).split(X, ye, groups), 1):
        model = LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1,
        )
        model.fit(X[tr], ye[tr])
        pred.append(model.predict(X[te]))
        true.append(ye[te])
        print(f"  fold {fold}/{n_splits} complete", flush=True)
    true, pred = np.concatenate(true), np.concatenate(pred)
    return {"accuracy": float(accuracy_score(true, pred)), "macro_f1": float(f1_score(true, pred, average="macro"))}


def main():
    df = load_data()
    channels = F.channel_list(use_back=False)
    results = {}
    for seconds, stride_seconds in [(1.0, 0.5), (2.0, 1.0), (3.0, 1.0)]:
        F.WINDOW = int(seconds * F.SAMPLE_RATE_HZ)
        F.STRIDE = int(stride_seconds * F.SAMPLE_RATE_HZ)
        t0 = time.time()
        print(f"[window] {seconds:.1f}s / stride {stride_seconds:.1f}s", flush=True)
        data = F.build_windows(df, channels)
        X = F.extract_window_features(data["windows"])
        result = evaluate(X, data["coarse"], data["dog_id"])
        result.update({"window_seconds": seconds, "stride_seconds": stride_seconds, "n_windows": int(len(X)), "elapsed_seconds": time.time() - t0})
        results[f"window_{seconds:g}s"] = result
        print(json.dumps(result, indent=2), flush=True)
    out = MODELS_DIR / "window_length_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
