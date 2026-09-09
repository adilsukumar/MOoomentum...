"""Fit focal-model embedding centroids and evaluate the complete quality gate."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import _LoadedModel, _DEVICE
from app.quality_gate import OODDetector, embed, run_quality_gate

CKPT = ROOT / "models" / "convnext_4class_focal" / "best.pt"
CENTROIDS = CKPT.parent / "quality_centroids.npz"
TEST_SET = ROOT / "models" / "quality_gate_test_set"
OUT = ROOT / "models" / "quality_gate_full_eval.json"


def fit_centroids(loaded):
    df = pd.read_csv(ROOT / "manifest.csv")
    df = df[(df.split == "train") & df.unified_label.isin(loaded.classes)]
    centroids, means, stds = [], [], []
    for cls in loaded.classes:
        vectors = []
        for path in df[df.unified_label == cls].filepath:
            with Image.open(path) as image:
                vectors.append(embed(loaded.model, loaded.transform, image, _DEVICE))
        vectors = np.stack(vectors)
        center = vectors.mean(0)
        distances = np.linalg.norm(vectors - center, axis=1)
        centroids.append(center)
        means.append(distances.mean())
        stds.append(distances.std())
    np.savez_compressed(CENTROIDS, classes=np.asarray(loaded.classes),
                        centroids=np.stack(centroids), within_class_mean=means,
                        within_class_std=stds)


def main():
    loaded = _LoadedModel(str(CKPT))
    if not CENTROIDS.exists():
        fit_centroids(loaded)
    detector = OODDetector(str(CENTROIDS))
    records = []
    for category in sorted(p for p in TEST_SET.iterdir() if p.is_dir()):
        expected_good = category.name == "good"
        for path in sorted(category.glob("*.jpg")):
            with Image.open(path) as image:
                gate = run_quality_gate(image.convert("RGB"), loaded.model,
                                        loaded.transform, _DEVICE, detector)
            records.append({"category": category.name, "file": path.name,
                            "expected_good": expected_good,
                            "passed": gate.passed, "reasons": gate.reasons,
                            "diagnostics": gate.to_dict()["diagnostics"]})
    good = [r for r in records if r["expected_good"]]
    bad = [r for r in records if not r["expected_good"]]
    fp = sum(not r["passed"] for r in good)
    fn = sum(r["passed"] for r in bad)
    report = {
        "checkpoint": str(CKPT), "centroids": str(CENTROIDS),
        "counts": {"good": len(good), "bad": len(bad), "false_positives": fp,
                   "false_negatives": fn},
        "rates": {"false_positive_rate": fp / len(good),
                  "false_negative_rate": fn / len(bad)},
        "records": records,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report["counts"], **report["rates"]}, indent=2))


if __name__ == "__main__":
    main()
