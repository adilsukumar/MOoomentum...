"""
Compute per-class embedding centroids (+ within-class spread) from a trained
checkpoint's TRAINING split, for the OOD detector in app/quality_gate.py.

Usage:
    python scripts/compute_centroids.py --ckpt models/convnext_4class/best.pt \
        --out models/convnext_4class/ood_centroids.npz --max-per-class 300
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import timm

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-class", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    manifest = pd.read_csv(ROOT / "manifest.csv")
    train_df = manifest[(manifest["split"] == "train") & (manifest["unified_label"].isin(classes))]

    centroids = []
    within_class_std = []
    print(f"Computing embeddings for up to {args.max_per_class} training images per class "
          f"({len(classes)} classes)...")
    for c in classes:
        sub = train_df[train_df["unified_label"] == c]
        if len(sub) > args.max_per_class:
            sub = sub.sample(n=args.max_per_class, random_state=42)
        embs = []
        for _, row in sub.iterrows():
            with Image.open(row["filepath"]) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                features = model.forward_features(x)
                pooled = model.forward_head(features, pre_logits=True)
            embs.append(pooled.float().cpu().numpy()[0])
        embs = np.stack(embs)
        centroid = embs.mean(axis=0)
        dists = np.linalg.norm(embs - centroid[None, :], axis=1)
        centroids.append(centroid)
        within_class_std.append(float(dists.std()))
        print(f"  {c:22s} n={len(sub):4d}  within-class dist std={dists.std():.3f}")

    np.savez(
        args.out,
        classes=np.array(classes),
        centroids=np.stack(centroids),
        within_class_std=np.array(within_class_std),
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
