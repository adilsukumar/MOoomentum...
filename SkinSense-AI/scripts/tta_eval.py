"""
Test-Time Augmentation (TTA) evaluation of the shipped ConvNeXt-Tiny model.

For each test image, run the model on: the plain center-crop view, a
horizontal flip, two off-center crops (top-left/bottom-right corners at a
slightly tighter scale), and a mild color-jitter variant. Average the
softmax probabilities across all views and compare against the single-pass
baseline already recorded in models/convnext/test_metrics.json.

No retraining involved -- this only changes inference-time behavior.

Usage:
    python scripts/tta_eval.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from torchvision import transforms
import timm
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_views(img_size):
    resize_to = int(round(img_size * 1.14))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def base(im):
        return transforms.functional.resize(im, resize_to)

    views = []

    # 1. plain center crop (== the existing single-pass eval transform)
    views.append(transforms.Compose([
        transforms.Resize(resize_to), transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ]))
    # 2. horizontal flip of the same center crop
    views.append(transforms.Compose([
        transforms.Resize(resize_to), transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), norm,
    ]))
    # 3 & 4. two off-center crops at a slightly tighter scale (top-left, bottom-right)
    tight = int(img_size * 0.92)

    def corner_crop(im, corner):
        im = transforms.functional.resize(im, int(round(tight * 1.14)))
        w, h = im.size
        if corner == "tl":
            box = (0, 0, tight, tight)
        else:
            box = (w - tight, h - tight, w, h)
        im = im.crop(box)
        return transforms.functional.resize(im, img_size)

    views.append(("corner_tl", tight))
    views.append(("corner_br", tight))

    # 5. mild color jitter on the plain center crop
    views.append(transforms.Compose([
        transforms.Resize(resize_to), transforms.CenterCrop(img_size),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), norm,
    ]))

    def apply(im, view):
        if isinstance(view, tuple):
            name, tight_size = view
            corner = "tl" if name == "corner_tl" else "br"
            cropped = corner_crop(im, corner)
            t = transforms.Compose([transforms.ToTensor(), norm])
            return t(cropped)
        else:
            return view(im)

    return views, apply


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / "models" / "convnext" / "best.pt", map_location=device, weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    manifest = pd.read_csv(ROOT / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    print(f"TTA-evaluating {len(test_df)} test images with 5 views each...")

    views, apply = build_views(img_size)
    label2idx = {c: i for i, c in enumerate(classes)}

    tta_probs = []
    single_probs = []
    labels = []
    for i, row in test_df.iterrows():
        with Image.open(row["filepath"]) as im:
            im = im.convert("RGB")
            batch = torch.stack([apply(im, v) for v in views]).to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(batch)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        single_probs.append(probs[0])          # view 0 = plain center crop = the existing baseline
        tta_probs.append(probs.mean(axis=0))   # mean across all 5 views
        labels.append(label2idx[row["unified_label"]])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test_df)}")

    labels = np.array(labels)
    single_probs = np.array(single_probs)
    tta_probs = np.array(tta_probs)

    def report(probs, name):
        pred = probs.argmax(1)
        acc = accuracy_score(labels, pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, pred, labels=list(range(len(classes))), zero_division=0
        )
        macro_f1 = f1.mean()
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
        for i, c in enumerate(classes):
            print(f"  {c:22s} precision={precision[i]:.3f}  recall={recall[i]:.3f}  f1={f1[i]:.3f}  support={support[i]}")
        return {
            "accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class": {classes[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(f1[i]), "support": int(support[i])} for i in range(len(classes))},
        }

    single_report = report(single_probs, "SINGLE-PASS (recomputed, should match test_metrics.json)")
    tta_report = report(tta_probs, "TTA (5-view average)")

    out = {"single_pass": single_report, "tta_5view": tta_report}
    (ROOT / "models" / "convnext" / "tta_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote models/convnext/tta_metrics.json")
    print(f"\nDelta: accuracy {tta_report['accuracy'] - single_report['accuracy']:+.4f}   "
          f"macro-F1 {tta_report['macro_f1'] - single_report['macro_f1']:+.4f}")


if __name__ == "__main__":
    main()
