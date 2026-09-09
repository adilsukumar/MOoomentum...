"""
Combine the EfficientNetV2 and ConvNeXt-Tiny test-set predictions into an ensemble
(simple mean of softmax probabilities) and report final metrics.

Usage:
    python scripts/ensemble_eval.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    d = np.load(ROOT / "models" / name / "test_probs.npz", allow_pickle=True)
    return d["probs"], d["labels"], list(d["classes"])


def main():
    probs_a, labels_a, classes_a = load("effnet")
    probs_b, labels_b, classes_b = load("convnext")

    assert classes_a == classes_b, f"class order mismatch: {classes_a} vs {classes_b}"
    assert len(labels_a) == len(labels_b), "test set size mismatch between models"
    assert np.array_equal(labels_a, labels_b), "test set label order mismatch between models"

    classes = classes_a
    y_true = labels_a

    ensemble_probs = (probs_a + probs_b) / 2.0
    y_pred_ens = ensemble_probs.argmax(1)
    y_pred_eff = probs_a.argmax(1)
    y_pred_cnx = probs_b.argmax(1)

    def report(y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0
        )
        macro_f1 = f1.mean()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
        for i, c in enumerate(classes):
            print(f"  {c:22s} precision={precision[i]:.3f}  recall={recall[i]:.3f}  "
                  f"f1={f1[i]:.3f}  support={support[i]}")
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_class": {
                classes[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                } for i in range(len(classes))
            },
            "confusion_matrix": cm.tolist(),
            "classes": classes,
        }

    eff_report = report(y_pred_eff, "EfficientNetV2 (solo)")
    cnx_report = report(y_pred_cnx, "ConvNeXt-Tiny (solo)")
    ens_report = report(y_pred_ens, "ENSEMBLE (mean of softmax probs)")

    out = {"efficientnetv2_solo": eff_report, "convnext_solo": cnx_report, "ensemble": ens_report}
    out_path = ROOT / "models" / "ensemble_test_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
