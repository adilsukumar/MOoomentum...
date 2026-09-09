"""
Aggregate results across the 5 repeated-split CV folds (item 8): reports
mean +/- std for Bacterial_dermatosis precision/recall/F1, and for the D7
high-confidence-accuracy claim (accuracy of predictions made at >=0.75
confidence, for Bacterial_dermatosis specifically vs. everything else).

Expects models/cv_fold{1..5}/test_metrics.json and test_probs.npz to exist
(each fold = original 5-class setup, weighted_ce, retrained from scratch with
a fresh --reseed group split).

Usage:
    python scripts/cv_analysis.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FOLDS = [f"cv_fold{i}" for i in range(1, 6)]
TARGET_CLASS = "Bacterial_dermatosis"


def main():
    bac_precision, bac_recall, bac_f1, bac_support = [], [], [], []
    overall_acc, overall_macro_f1 = [], []
    hc_acc_bacterial, hc_n_bacterial = [], []
    hc_acc_other, hc_n_other = [], []
    hc_correct_bacterial, hc_correct_other = [], []
    per_fold = []

    for fold in FOLDS:
        metrics_path = ROOT / "models" / fold / "test_metrics.json"
        probs_path = ROOT / "models" / fold / "test_probs.npz"
        if not metrics_path.exists():
            print(f"MISSING: {metrics_path} -- skipping {fold}")
            continue
        m = json.loads(metrics_path.read_text())
        bac = m["per_class"][TARGET_CLASS]
        bac_precision.append(bac["precision"])
        bac_recall.append(bac["recall"])
        bac_f1.append(bac["f1"])
        bac_support.append(bac["support"])
        overall_acc.append(m["accuracy"])
        overall_macro_f1.append(m["macro_f1"])

        d = np.load(probs_path, allow_pickle=True)
        probs, labels, classes = d["probs"], d["labels"], list(d["classes"])
        bac_idx = classes.index(TARGET_CLASS)
        pred = probs.argmax(1)
        conf = probs.max(1)

        is_bac_pred = pred == bac_idx
        hc_mask_bac = is_bac_pred & (conf >= 0.75)
        hc_mask_other = (~is_bac_pred) & (conf >= 0.75)
        correct = pred == labels

        # Keep one entry per fold even when no prediction clears the threshold.
        # Omitting zero-count folds silently misaligns n_per_fold with the
        # accuracy arrays and makes an unstable small-n claim look cleaner.
        n_bac = int(hc_mask_bac.sum())
        n_other = int(hc_mask_other.sum())
        hc_acc_bacterial.append(
            float(correct[hc_mask_bac].mean()) if n_bac else None
        )
        hc_n_bacterial.append(n_bac)
        hc_correct_bacterial.append(int(correct[hc_mask_bac].sum()))
        hc_acc_other.append(
            float(correct[hc_mask_other].mean()) if n_other else None
        )
        hc_n_other.append(n_other)
        hc_correct_other.append(int(correct[hc_mask_other].sum()))
        per_fold.append({
            "fold": fold,
            "overall_accuracy": m["accuracy"],
            "overall_macro_f1": m["macro_f1"],
            "bacterial_precision": bac["precision"],
            "bacterial_recall": bac["recall"],
            "bacterial_f1": bac["f1"],
            "bacterial_support": bac["support"],
            "high_conf_bacterial_n": n_bac,
            "high_conf_bacterial_correct": int(correct[hc_mask_bac].sum()),
            "high_conf_bacterial_accuracy": hc_acc_bacterial[-1],
        })

        print(f"{fold}: overall acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}  "
              f"Bacterial_dermatosis P={bac['precision']:.3f} R={bac['recall']:.3f} F1={bac['f1']:.3f} "
              f"(support={bac['support']})  "
              f"high-conf(>=0.75) Bacterial preds: n={int(hc_mask_bac.sum())} "
              f"acc={'%.3f' % correct[hc_mask_bac].mean() if hc_mask_bac.sum() else 'n/a'}")

    def mstd(xs):
        observed = [x for x in xs if x is not None]
        return (float(np.mean(observed)), float(np.std(observed))) if observed else (None, None)

    result = {
        "n_folds_completed": len(overall_acc),
        "per_fold": per_fold,
        "overall_accuracy": dict(zip(["mean", "std"], mstd(overall_acc))),
        "overall_macro_f1": dict(zip(["mean", "std"], mstd(overall_macro_f1))),
        "bacterial_dermatosis": {
            "precision": dict(zip(["mean", "std"], mstd(bac_precision))),
            "recall": dict(zip(["mean", "std"], mstd(bac_recall))),
            "f1": dict(zip(["mean", "std"], mstd(bac_f1))),
            "test_support_per_fold": bac_support,
        },
        "high_confidence_accuracy_bacterial_predictions": {
            "mean": mstd(hc_acc_bacterial)[0], "std": mstd(hc_acc_bacterial)[1],
            "n_per_fold": hc_n_bacterial,
            "correct_per_fold": hc_correct_bacterial,
            "pooled_accuracy": (sum(hc_correct_bacterial) / sum(hc_n_bacterial)
                                if sum(hc_n_bacterial) else None),
        },
        "high_confidence_accuracy_other_predictions": {
            "mean": mstd(hc_acc_other)[0], "std": mstd(hc_acc_other)[1],
            "n_per_fold": hc_n_other,
            "correct_per_fold": hc_correct_other,
            "pooled_accuracy": (sum(hc_correct_other) / sum(hc_n_other)
                                if sum(hc_n_other) else None),
        },
    }

    print("\n=== SUMMARY ACROSS FOLDS ===")
    print(f"Overall accuracy: {result['overall_accuracy']['mean']:.4f} +/- {result['overall_accuracy']['std']:.4f}")
    print(f"Overall macro-F1: {result['overall_macro_f1']['mean']:.4f} +/- {result['overall_macro_f1']['std']:.4f}")
    bd = result["bacterial_dermatosis"]
    print(f"Bacterial_dermatosis precision: {bd['precision']['mean']:.4f} +/- {bd['precision']['std']:.4f}")
    print(f"Bacterial_dermatosis recall:    {bd['recall']['mean']:.4f} +/- {bd['recall']['std']:.4f}")
    print(f"Bacterial_dermatosis F1:        {bd['f1']['mean']:.4f} +/- {bd['f1']['std']:.4f}")
    print(f"Bacterial_dermatosis test support per fold: {bd['test_support_per_fold']}")
    hcb = result["high_confidence_accuracy_bacterial_predictions"]
    print(f"\nHigh-confidence (>=0.75) Bacterial_dermatosis prediction accuracy: "
          f"{hcb['mean']} +/- {hcb['std']}  (n per fold: {hcb['n_per_fold']})")
    hco = result["high_confidence_accuracy_other_predictions"]
    print(f"High-confidence (>=0.75) OTHER-class prediction accuracy: "
          f"{hco['mean']} +/- {hco['std']}  (n per fold: {hco['n_per_fold']})")

    (ROOT / "models" / "cv_analysis.json").write_text(json.dumps(result, indent=2))
    print("\nWrote models/cv_analysis.json")


if __name__ == "__main__":
    main()
