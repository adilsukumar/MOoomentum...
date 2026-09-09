"""
Construct a small good/bad image test set for evaluating app/quality_gate.py
(PROJECT.md item 9). No such test set existed, so this builds one from real
dataset images plus synthetic degradations.

Categories:
  good/          -- 15 real, clean test-set images (should PASS the gate)
  bad_blurry/    -- the same 15 images, heavily Gaussian-blurred (should FAIL: blur)
  bad_tiny/      -- 10 images downsized to 32x32 (should FAIL: framing/size)
  bad_flat/      -- 10 solid-color images (should FAIL: framing/blank)
  bad_noise/     -- 10 pure random-noise images (should FAIL: OOD -- wrong subject)
  bad_aspect/    -- 10 images stretched to an extreme aspect ratio (should FAIL: framing)

Usage:
    python scripts/build_quality_test_set.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "quality_gate_test_set"


def main():
    random.seed(11)
    np.random.seed(11)
    for sub in ["good", "bad_blurry", "bad_tiny", "bad_flat", "bad_noise", "bad_aspect"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].sample(n=15, random_state=11)

    for i, (_, row) in enumerate(test_df.iterrows()):
        with Image.open(row["filepath"]) as im:
            im = im.convert("RGB")
            im.save(OUT / "good" / f"good_{i:02d}.jpg")
            blurry = im.filter(ImageFilter.GaussianBlur(radius=12))
            blurry.save(OUT / "bad_blurry" / f"blurry_{i:02d}.jpg")

    for i in range(10):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).resize((32, 32)).save(OUT / "bad_tiny" / f"tiny_{i:02d}.jpg")

    flat_colors = [(200, 200, 200), (30, 30, 30), (255, 255, 255), (0, 0, 0), (128, 64, 64),
                   (64, 128, 64), (64, 64, 128), (180, 150, 100), (100, 180, 150), (150, 100, 180)]
    for i, color in enumerate(flat_colors):
        Image.new("RGB", (300, 300), color).save(OUT / "bad_flat" / f"flat_{i:02d}.jpg")

    for i in range(10):
        arr = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(OUT / "bad_noise" / f"noise_{i:02d}.jpg")

    for i, (_, row) in enumerate(test_df.head(10).iterrows()):
        with Image.open(row["filepath"]) as im:
            im = im.convert("RGB").resize((600, 40))  # extreme 15:1 aspect ratio
            im.save(OUT / "bad_aspect" / f"aspect_{i:02d}.jpg")

    print(f"Wrote quality-gate test set under {OUT}")
    for sub in ["good", "bad_blurry", "bad_tiny", "bad_flat", "bad_noise", "bad_aspect"]:
        n = len(list((OUT / sub).glob("*.jpg")))
        print(f"  {sub:12s} {n} images")


if __name__ == "__main__":
    main()
