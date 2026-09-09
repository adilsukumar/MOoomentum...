"""
Item 8: raw-signal 1D-CNN baseline, no hand-crafted features — since a
Keras model gives a clean TFLite export path, this doubles as a real
export-format candidate, not just an accuracy comparison.

Architecture (small, ~Palaz-et-al scale): 2 conv blocks + global average
pooling + 2 dense layers. Trained directly on the windowed raw neck-sensor
signal (8 channels: 6 raw axes + 2 magnitude), same LODO GroupKFold split
as the feature+tree pipeline, coarse (3-class) target only — matching the
deployable target the rest of this task list optimizes for.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from features import add_magnitude, build_windows, channel_list, NECK_AXES, BACK_AXES, VALID_BEHAVIORS, COARSE_MAP

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "DogMoveData.csv"
MODELS_DIR = ROOT / "models"
N_FOLDS = 5
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 256

tf.random.set_seed(RANDOM_STATE)


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


def build_model(n_timesteps, n_channels, n_classes):
    inputs = tf.keras.Input(shape=(n_timesteps, n_channels))
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = load_data()
    neck_channels = channel_list(use_back=False)
    print("[window] building neck-only windows...")
    data = build_windows(df, neck_channels)
    windows = data["windows"]  # (n, 100, 8) raw signal, not hand-crafted features
    dog_id = data["dog_id"]
    y_coarse = data["coarse"]
    print(f"[window] {len(windows):,} windows, shape per window {windows.shape[1:]}")

    # per-channel standardization (fit on training fold only, applied to test)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_coarse)
    n_classes = len(le.classes_)

    class_counts = np.bincount(y_enc)
    class_weight = {i: len(y_enc) / (n_classes * c) for i, c in enumerate(class_counts)}

    gkf = GroupKFold(n_splits=N_FOLDS)
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(gkf.split(windows, y_enc, dog_id)):
        t0 = time.time()
        mu = windows[tr].mean(axis=(0, 1), keepdims=True)
        sigma = windows[tr].std(axis=(0, 1), keepdims=True) + 1e-6
        X_tr = (windows[tr] - mu) / sigma
        X_te = (windows[te] - mu) / sigma

        model = build_model(windows.shape[1], windows.shape[2], n_classes)
        model.fit(
            X_tr, y_enc[tr], validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=class_weight, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )
        pred = model.predict(X_te, verbose=0).argmax(axis=1)
        y_true_all.append(y_enc[te])
        y_pred_all.append(pred)
        print(f"[fold {fold+1}/{N_FOLDS}] done in {time.time()-t0:.1f}s, "
              f"held out {len(set(dog_id[te]))} dogs")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    report = classification_report(
        le.inverse_transform(y_true_all), le.inverse_transform(y_pred_all),
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        le.inverse_transform(y_true_all), le.inverse_transform(y_pred_all), zero_division=0))

    with open(MODELS_DIR / "cnn_results.json", "w") as f:
        json.dump({"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
                    "per_class": {k: v for k, v in report.items() if k in ("Active", "Resting", "Other")}},
                   f, indent=2)
    print(f"\nSaved cnn_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
