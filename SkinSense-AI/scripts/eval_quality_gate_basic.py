"""
Evaluate the blur + framing checks of app/quality_gate.py against the
constructed test set (scripts/build_quality_test_set.py), WITHOUT the OOD
component (that needs a trained model + centroids, added once the final
model from the retraining experiments is chosen -- see
scripts/eval_quality_gate_full.py).

Usage:
    python scripts/eval_quality_gate_basic.py
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from app.quality_gate import run_quality_gate

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "models" / "quality_gate_test_set"

EXPECT_PASS = {"good"}
EXPECT_FAIL = {"bad_blurry", "bad_tiny", "bad_flat", "bad_aspect"}
# bad_noise is intentionally excluded here -- catching "wrong subject" noise
# images is the OOD check's job, not blur/framing; see eval_quality_gate_full.py


def main():
    results = {}
    fp, fn, tp, tn = 0, 0, 0, 0
    for category in sorted(TEST_SET.iterdir()):
        if not category.is_dir():
            continue
        expected_pass = category.name in EXPECT_PASS
        cat_results = []
        for img_path in sorted(category.glob("*.jpg")):
            with Image.open(img_path) as im:
                gate = run_quality_gate(im.convert("RGB"))
            cat_results.append({"file": img_path.name, "passed": gate.passed, "reasons": gate.reasons})
            if category.name in EXPECT_PASS or category.name in EXPECT_FAIL:
                if expected_pass and not gate.passed:
                    fp += 1
                elif expected_pass and gate.passed:
                    tn += 1
                elif not expected_pass and gate.passed:
                    fn += 1
                elif not expected_pass and not gate.passed:
                    tp += 1
        results[category.name] = cat_results
        n_passed = sum(1 for r in cat_results if r["passed"])
        print(f"{category.name:12s} {n_passed}/{len(cat_results)} passed the gate")
        if category.name != "good":
            for r in cat_results:
                if r["passed"]:
                    print(f"    MISSED (should have failed): {r['file']}")

    print(f"\nOn categories with a known expectation (good / bad_blurry / bad_tiny / bad_flat / bad_aspect):")
    print(f"  False positives (good image wrongly rejected): {fp}")
    print(f"  False negatives (bad image wrongly passed):    {fn}")
    print(f"  True positives (bad image correctly rejected): {tp}")
    print(f"  True negatives (good image correctly passed):  {tn}")

    (ROOT / "models" / "quality_gate_basic_eval.json").write_text(json.dumps({
        "results": results,
        "false_positives": fp, "false_negatives": fn, "true_positives": tp, "true_negatives": tn,
    }, indent=2))
    print("\nWrote models/quality_gate_basic_eval.json")
    print("\nNote: bad_noise (wrong-subject test) is NOT scored here -- blur/framing checks aren't "
          "meant to catch it; that's the OOD centroid check's job, evaluated separately once a "
          "final model is chosen (needs trained-model embeddings).")


if __name__ == "__main__":
    main()
