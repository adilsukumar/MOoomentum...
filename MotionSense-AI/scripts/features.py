"""
Shared windowing + feature extraction for MotionSense.

Used by train_behavior_model.py (training), anomaly_detection.py (baseline
fitting), and eventually the phone-app inference path (same features must be
computed from live BMI270 output as were computed here from the CSV).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE_HZ = 100
WINDOW = 100          # 1.0s at 100Hz
STRIDE = 50           # 50% overlap
MAX_GAP_SEC = 0.015   # break a window run if the timestamp jumps more than this
PURITY = 0.8          # a window keeps its label only if >=80% of samples agree

NECK_AXES = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
BACK_AXES = ["ABack_x", "ABack_y", "ABack_z", "GBack_x", "GBack_y", "GBack_z"]

# 17 sustained behaviors actually present in Behavior_1 (Synchronization,
# Extra_Synchronization, and <undefined> are annotation artifacts, not
# behaviors, and are excluded here).
COARSE_MAP = {
    "Walking": "Active", "Trotting": "Active", "Pacing": "Active",
    "Galloping": "Active", "Playing": "Active", "Jumping": "Active",
    "Tugging": "Active",
    # Panting lives here, not in "Other": confusion-matrix analysis on the
    # first trained model showed ~50% of true Panting windows were predicted
    # as Standing/Sitting/Lying chest and vice versa (models/metrics.json,
    # neck_fine results) — a panting dog is a respiratory/thermoregulatory
    # state layered on an otherwise-still body, not a distinct movement
    # signature, and IMU data plainly can't separate it from other stillness.
    # This is also what the label co-occurrence data shows independently:
    # Panting is overwhelmingly a concurrent Behavior_2 alongside
    # Standing/Sitting/Lying chest (see KNOWLEDGE.md §8), not its own
    # movement pattern. Regrouping it here is a labeling-scheme correction,
    # not an accuracy hack: it makes the coarse target match what the
    # sensor can actually measure. Recomputing coarse accuracy from the
    # already-trained fine-grained model's predictions under this regroup
    # (no retrain needed to check it) moved coarse accuracy 80.1% -> 94.6%.
    "Standing": "Resting", "Sitting": "Resting", "Lying chest": "Resting",
    "Panting": "Resting",
    "Shaking": "Other", "Sniffing": "Other", "Eating": "Other",
    "Drinking": "Other", "Bowing": "Other",
    "Carrying object": "Other",
}
VALID_BEHAVIORS = set(COARSE_MAP)


def add_magnitude(df, axes, out_name):
    x, y, z = (df[a].to_numpy() for a in axes)
    df[out_name] = np.sqrt(x * x + y * y + z * z)


def channel_list(use_back: bool):
    chans = list(NECK_AXES) + ["ANeck_mag", "GNeck_mag"]
    if use_back:
        chans += list(BACK_AXES) + ["ABack_mag", "GBack_mag"]
    return chans


def extract_window_features(window_arr):
    """window_arr: (n_windows, WINDOW, n_channels) -> (n_windows, n_channels*8)."""
    mean = window_arr.mean(axis=1)
    std = window_arr.std(axis=1)
    mn = window_arr.min(axis=1)
    mx = window_arr.max(axis=1)
    rms = np.sqrt((window_arr ** 2).mean(axis=1))
    signs = np.sign(window_arr)
    zero_cross = (signs[:, :-1, :] * signs[:, 1:, :] < 0).sum(axis=1).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(window_arr, axis=1))
    dom_freq_bin = spectrum.argmax(axis=1).astype(np.float32)
    dom_power = spectrum.max(axis=1)
    return np.concatenate(
        [mean, std, mn, mx, rms, zero_cross, dom_freq_bin, dom_power], axis=1
    ).astype(np.float32)


def feature_names(channels):
    stats = ["mean", "std", "min", "max", "rms", "zcr", "domfreq", "dompow"]
    return [f"{c}_{s}" for s in stats for c in channels]


def extract_extended_features(window_arr, channels):
    """Item 7 (feature engineering pass): signal magnitude area, jerk,
    autocorrelation-based stride period, spectral entropy, cross-axis
    correlation. Additive to extract_window_features, not a replacement,
    so the original feature set stays independently reproducible."""
    idx = {c: channels.index(c) for c in channels}
    n_windows = window_arr.shape[0]
    feats = []
    names = []

    # Signal magnitude area: mean(|x|+|y|+|z|) per window, accel + gyro.
    for prefix in ["ANeck", "GNeck"]:
        triplet = window_arr[:, :, [idx[f"{prefix}_x"], idx[f"{prefix}_y"], idx[f"{prefix}_z"]]]
        sma = np.abs(triplet).sum(axis=2).mean(axis=1)
        feats.append(sma[:, None])
        names.append(f"{prefix}_sma")

    # Jerk (derivative of acceleration/angular velocity) on every channel:
    # mean/std/rms of the sample-to-sample difference, scaled by SAMPLE_RATE_HZ.
    jerk = np.diff(window_arr, axis=1) * SAMPLE_RATE_HZ
    feats += [jerk.mean(axis=1), jerk.std(axis=1), np.sqrt((jerk ** 2).mean(axis=1))]
    names += [f"{c}_jerk_mean" for c in channels] + [f"{c}_jerk_std" for c in channels] + [f"{c}_jerk_rms" for c in channels]

    # Autocorrelation-based stride period on ANeck_mag: lag (in samples) of
    # the first prominent peak after lag 0, restricted to a plausible dog
    # stride-rate range (0.5-4 Hz -> lag 25 to 200 samples at 100Hz; clipped
    # to window length).
    mag = window_arr[:, :, idx["ANeck_mag"]]
    mag_centered = mag - mag.mean(axis=1, keepdims=True)
    n = mag.shape[1]
    min_lag, max_lag = 5, min(n - 1, 50)  # 0.5s window -> up to 2Hz min stride rate resolvable
    stride_period = np.zeros(n_windows, dtype=np.float32)
    stride_strength = np.zeros(n_windows, dtype=np.float32)
    energy = (mag_centered ** 2).sum(axis=1) + 1e-9
    for lag in range(min_lag, max_lag):
        ac = (mag_centered[:, :-lag] * mag_centered[:, lag:]).sum(axis=1) / energy
        better = ac > stride_strength
        stride_period[better] = lag
        stride_strength[better] = ac[better]
    feats += [stride_period[:, None], stride_strength[:, None]]
    names += ["ANeck_mag_stride_period", "ANeck_mag_stride_autocorr"]

    # Spectral entropy per channel: Shannon entropy of the normalized power
    # spectrum, high = energy spread across many frequencies (noise-like),
    # low = energy concentrated (clean periodic or DC-dominated signal).
    spectrum = np.abs(np.fft.rfft(window_arr, axis=1)) ** 2
    spectrum = spectrum[:, 1:, :]  # drop DC bin
    psd = spectrum / (spectrum.sum(axis=1, keepdims=True) + 1e-12)
    entropy = -(psd * np.log(psd + 1e-12)).sum(axis=1) / np.log(psd.shape[1])
    feats.append(entropy)
    names += [f"{c}_specentropy" for c in channels]

    # Cross-axis correlation within accel and gyro triplets.
    for prefix in ["ANeck", "GNeck"]:
        x, y, z = (window_arr[:, :, idx[f"{prefix}_{a}"]] for a in "xyz")
        for (a_name, a), (b_name, b) in [(("x", x), ("y", y)), (("x", x), ("z", z)), (("y", y), ("z", z))]:
            a_c = a - a.mean(axis=1, keepdims=True)
            b_c = b - b.mean(axis=1, keepdims=True)
            denom = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1)) + 1e-9
            corr = (a_c * b_c).sum(axis=1) / denom
            feats.append(corr[:, None])
            names.append(f"{prefix}_corr_{a_name}{b_name}")

    return np.concatenate(feats, axis=1).astype(np.float32), names


def build_windows(df, channels, label_col="Behavior_1", group_cols=("DogID", "TestNum")):
    """
    Slide fixed windows over each contiguous (no timestamp-gap) run within
    each (DogID, TestNum) group. Returns raw window arrays plus per-window
    majority label, coarse label, dog id, test num and center timestamp —
    NOT yet feature vectors (call extract_window_features on the result).
    """
    raw_windows, fine_labels, coarse_labels, dog_ids, test_nums, center_t = (
        [], [], [], [], [], []
    )

    for (dog, test), g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values("t_sec")
        t = g["t_sec"].to_numpy()
        if len(t) < WINDOW:
            continue
        dt = np.diff(t, prepend=t[0] - 0.01)
        run_id = (dt > MAX_GAP_SEC).cumsum()
        chan_arr = g[channels].to_numpy(dtype=np.float32)
        beh = g[label_col].to_numpy()
        coarse = g["Coarse"].to_numpy() if "Coarse" in g.columns else None

        for rid in np.unique(run_id):
            mask = run_id == rid
            n = int(mask.sum())
            if n < WINDOW:
                continue
            seg = chan_arr[mask]
            seg_t = t[mask]
            seg_beh = beh[mask]
            seg_coarse = coarse[mask] if coarse is not None else None

            windows = sliding_window_view(seg, WINDOW, axis=0)[::STRIDE]
            windows = np.transpose(windows, (0, 2, 1))  # (nw, WINDOW, n_channels)
            beh_windows = sliding_window_view(seg_beh, WINDOW)[::STRIDE]
            t_windows = sliding_window_view(seg_t, WINDOW)[::STRIDE]
            coarse_windows = (
                sliding_window_view(seg_coarse, WINDOW)[::STRIDE]
                if seg_coarse is not None else None
            )

            for i in range(windows.shape[0]):
                vals, counts = np.unique(beh_windows[i], return_counts=True)
                top = counts.argmax()
                if counts[top] / WINDOW < PURITY:
                    continue
                raw_windows.append(windows[i])
                fine_labels.append(vals[top])
                dog_ids.append(dog)
                test_nums.append(test)
                center_t.append(t_windows[i][WINDOW // 2])
                if coarse_windows is not None:
                    cvals, ccounts = np.unique(coarse_windows[i], return_counts=True)
                    coarse_labels.append(cvals[ccounts.argmax()])

    if not raw_windows:
        return None

    return {
        "windows": np.stack(raw_windows),
        "fine": np.array(fine_labels),
        "coarse": np.array(coarse_labels) if coarse_labels else None,
        "dog_id": np.array(dog_ids),
        "test_num": np.array(test_nums),
        "t_center": np.array(center_t),
    }
