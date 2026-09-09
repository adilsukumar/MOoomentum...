"""
Consumes models/oof_predictions.joblib (see oof_export.py) to run three
cheap post-hoc experiments without any retraining:

  1. Temporal smoothing (majority vote over the last N raw predictions)
  2. Confidence-threshold coverage/accuracy tradeoff
  3. Hierarchical gating for fine-grained output

Prints a report for each; also writes models/postprocess_results.json.
"""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_oof():
    return joblib.load(MODELS_DIR / "oof_predictions.joblib")


def temporal_smoothing(o, n_values=(5, 10)):
    df = pd.DataFrame({
        "dog_id": o["dog_id"], "test_num": o["test_num"], "t_center": o["t_center"],
        "y": o["y_coarse"], "pred": o["pred_coarse"],
    })
    baseline_acc = accuracy_score(df["y"], df["pred"])
    baseline_report = classification_report(df["y"], df["pred"], output_dict=True, zero_division=0)

    results = {"baseline": {"accuracy": baseline_acc, "macro_f1": baseline_report["macro avg"]["f1-score"]}}

    for n in n_values:
        smoothed = np.empty(len(df), dtype=object)
        for (dog, test), g in df.groupby(["dog_id", "test_num"], sort=False):
            g = g.sort_values("t_center")
            preds = g["pred"].to_numpy()
            sm = np.empty(len(preds), dtype=object)
            for i in range(len(preds)):
                window = preds[max(0, i - n + 1): i + 1]
                sm[i] = Counter(window).most_common(1)[0][0]
            smoothed[g.index.to_numpy()] = sm
        acc = accuracy_score(df["y"], smoothed)
        report = classification_report(df["y"], smoothed, output_dict=True, zero_division=0)
        results[f"smoothed_N{n}"] = {"accuracy": acc, "macro_f1": report["macro avg"]["f1-score"],
                                      "per_class": {k: report[k] for k in report if k in ("Active", "Resting", "Other")}}
        print(f"[smoothing N={n}] accuracy={acc:.4f} macro_f1={report['macro avg']['f1-score']:.4f} "
              f"(baseline accuracy={baseline_acc:.4f} macro_f1={baseline_report['macro avg']['f1-score']:.4f})")

    return results


def confidence_threshold(o, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    y = o["y_coarse"]
    pred = o["pred_coarse"]
    proba = o["proba_coarse_max"]
    n = len(y)
    results = {}
    for t in thresholds:
        confident = proba >= t
        coverage = confident.mean()
        if confident.sum() == 0:
            acc = None
        else:
            acc = accuracy_score(y[confident], pred[confident])
        results[str(t)] = {"coverage": float(coverage), "accuracy_on_confident": acc}
        print(f"[confidence>={t}] coverage={coverage:.3f} accuracy_on_confident="
              f"{acc if acc is None else round(acc,4)}")
    return results


def hierarchical_gating(o, fine_thresholds=(0.4, 0.5, 0.6, 0.7)):
    y_coarse = o["y_coarse"]
    pred_coarse = o["pred_coarse"]
    active_mask = o["active_mask"]

    # Map full-length arrays for the active-only fine sub-model back into
    # the full window index space.
    n = len(y_coarse)
    pred_fine_active_full = np.empty(n, dtype=object)
    proba_fine_active_full = np.zeros(n, dtype=np.float32)
    pred_fine_active_full[active_mask] = o["pred_fine_active"]
    proba_fine_active_full[active_mask] = o["proba_fine_active_max"]

    y_fine = o["y_fine"]  # true specific behavior, always available for scoring

    results = {}
    for t in fine_thresholds:
        effective_pred = np.empty(n, dtype=object)
        effective_true = np.empty(n, dtype=object)
        for i in range(n):
            effective_true[i] = y_fine[i] if y_coarse[i] == "Active" else y_coarse[i]
            if pred_coarse[i] == "Active":
                if proba_fine_active_full[i] >= t:
                    effective_pred[i] = pred_fine_active_full[i]
                else:
                    effective_pred[i] = "Active (unspecified)"
            else:
                effective_pred[i] = pred_coarse[i]
        acc = accuracy_score(effective_true, effective_pred)
        # what fraction of true-Active windows got a specific label vs "unspecified"
        true_active = y_coarse == "Active"
        specified_rate = (effective_pred[true_active] != "Active (unspecified)").mean()
        results[str(t)] = {"effective_accuracy": float(acc), "active_specified_rate": float(specified_rate)}
        print(f"[hierarchical gate, fine_threshold={t}] effective_accuracy={acc:.4f} "
              f"active_windows_given_specific_label={specified_rate:.3f}")

    # also report the plain (ungated) full 17-class accuracy for comparison
    ungated_acc = accuracy_score(o["y_fine"], o["pred_fine"])
    print(f"[reference] plain ungated 17-class accuracy (from oof_export's full fine model): {ungated_acc:.4f}")
    results["ungated_17class_reference_accuracy"] = float(ungated_acc)
    return results


def main():
    o = load_oof()
    print("=== 1. Temporal smoothing ===")
    smoothing_results = temporal_smoothing(o)
    print("\n=== 2. Confidence-aware output ===")
    confidence_results = confidence_threshold(o)
    print("\n=== 3. Hierarchical gating ===")
    gating_results = hierarchical_gating(o)

    with open(MODELS_DIR / "postprocess_results.json", "w") as f:
        json.dump({
            "temporal_smoothing": smoothing_results,
            "confidence_threshold": confidence_results,
            "hierarchical_gating": gating_results,
        }, f, indent=2)
    print(f"\nSaved postprocess_results.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
