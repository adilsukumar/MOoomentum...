"""
Break out the shipped model's test-set performance by ORIGINAL pre-merge label,
within each unified class (PROJECT.md decision D1 merged:
  Allergic_dermatitis  <- Dermatitis, Hypersensitivity, Hypersensitivity_allergic_dermatosis
  Fungal_infection     <- Fungal_infections, ringworm
This checks whether one sub-label is dragging its merged class's metrics down
disproportionately -- if so, that's a signal the merge itself needs revisiting,
not just the model.

Usage:
    python scripts/audit_sublabels.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / "models" / "convnext" / "best.pt", map_location=device, weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    idx2label = {i: c for i, c in enumerate(classes)}
    label2idx = {c: i for i, c in enumerate(classes)}

    tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    manifest = pd.read_csv(ROOT / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    print(f"Running inference on {len(test_df)} test images for sub-label breakdown...")

    rows = []
    for i, row in test_df.iterrows():
        with Image.open(row["filepath"]) as im:
            im = im.convert("RGB")
            x = tf(im).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
        pred_idx = logits.argmax(1).item()
        conf = torch.softmax(logits.float(), 1).max().item()
        rows.append({
            "raw_label": row["raw_label"],
            "unified_label": row["unified_label"],
            "predicted": idx2label[pred_idx],
            "confidence": conf,
            "correct": idx2label[pred_idx] == row["unified_label"],
        })
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(test_df)}")

    df = pd.DataFrame(rows)

    print("\n=== Per-original-label accuracy, grouped by merged unified class ===")
    summary = {}
    for unified in sorted(df["unified_label"].unique()):
        sub = df[df["unified_label"] == unified]
        raw_labels_in_group = sorted(sub["raw_label"].unique())
        if len(raw_labels_in_group) <= 1:
            continue  # not a merged class, nothing to break out
        print(f"\n{unified}  (overall n={len(sub)}, overall acc={sub['correct'].mean():.4f})")
        summary[unified] = {}
        for raw in raw_labels_in_group:
            rs = sub[sub["raw_label"] == raw]
            acc = rs["correct"].mean()
            mean_conf = rs["confidence"].mean()
            # where do the wrong ones actually go?
            wrong = rs[~rs["correct"]]
            misroute = wrong["predicted"].value_counts().to_dict()
            print(f"  raw_label={raw:20s} n={len(rs):4d}  accuracy={acc:.4f}  mean_confidence={mean_conf:.3f}  "
                  f"misroutes={misroute}")
            summary[unified][raw] = {
                "n": int(len(rs)), "accuracy": float(acc), "mean_confidence": float(mean_conf),
                "misroutes": {k: int(v) for k, v in misroute.items()},
            }

    (ROOT / "models" / "sublabel_audit.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote models/sublabel_audit.json")


if __name__ == "__main__":
    main()
