"""
Fine-tune a pretrained timm backbone on the SkinSense manifest (manifest.csv).

Usage:
    python scripts/train.py --arch convnext_tiny --out models/convnext --epochs 18
    python scripts/train.py --arch convnext_tiny --out models/convnext_4class --epochs 18 \
        --exclude-classes Bacterial_dermatosis
    python scripts/train.py --arch convnext_tiny --out models/convnext_focal --epochs 18 \
        --exclude-classes Bacterial_dermatosis --loss focal
    python scripts/train.py --arch convnext_tiny --out models/convnext_384 --epochs 18 \
        --exclude-classes Bacterial_dermatosis --img-size 384 --grad-checkpointing --batch-size 16
    python scripts/train.py --arch convnext_tiny --out models/cv_fold1 --epochs 15 \
        --reseed 101   # ignores manifest's stored split column, regenerates a fresh group split

Writes:
    <out>/best.pt          - checkpoint: state_dict, arch, classes, img_size
    <out>/history.json     - per-epoch train/val loss & macro-F1
    <out>/test_metrics.json - final test-set accuracy, macro-F1, per-class P/R/F1, confusion matrix
    <out>/test_probs.npz   - per-test-image softmax probs, labels, filepaths (for later analysis/ensembling)
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SkinDataset(Dataset):
    def __init__(self, df, classes, img_size, train: bool):
        self.paths = df["filepath"].tolist()
        self.label2idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.label2idx[l] for l in df["unified_label"].tolist()]
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            return self.tf(im), self.labels[idx]

    def __getitem__(self, idx):
        try:
            return self._load(idx)
        except Exception:
            # corrupt/unreadable image: fall back to a different random sample
            return self._load((idx + 1) % len(self.paths))


def regroup_split(df_all, seed):
    """
    Re-derive a fresh 70/15/15 group-stratified split from scratch, ignoring the
    manifest's stored 'split' column. Mirrors scripts/prepare_data.py's grouping
    logic exactly (same group_id column, same majority-label-per-group stratum,
    same ratios) but with a caller-supplied seed instead of the fixed seed=42
    used for the shipped model's split. Used for repeated-split robustness
    checks (see PROJECT.md item 8) -- NOT used by default, so the shipped
    model's reproducible split is untouched unless --reseed is passed.
    """
    rng = random.Random(seed)
    group_to_indices = defaultdict(list)
    for i, row in df_all.iterrows():
        group_to_indices[row["group_id"]].append(i)

    group_label = {}
    for g, idxs in group_to_indices.items():
        labels = [df_all.at[i, "unified_label"] for i in idxs]
        group_label[g] = max(set(labels), key=labels.count)

    labels_to_groups = defaultdict(list)
    for g, lbl in group_label.items():
        labels_to_groups[lbl].append(g)

    split_of_group = {}
    for lbl, groups in labels_to_groups.items():
        groups = groups[:]
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for g in groups[:n_train]:
            split_of_group[g] = "train"
        for g in groups[n_train:n_train + n_val]:
            split_of_group[g] = "val"
        for g in groups[n_train + n_val:]:
            split_of_group[g] = "test"

    df_all = df_all.copy()
    df_all["split"] = df_all["group_id"].map(split_of_group)
    return df_all


def build_loaders(manifest_path, classes, img_size, batch_size, num_workers, reseed=None):
    df = pd.read_csv(manifest_path)
    if reseed is not None:
        df = regroup_split(df, reseed)
    df = df[df["unified_label"].isin(classes)]
    loaders = {}
    for split, train_flag in [("train", True), ("val", False), ("test", False)]:
        sub = df[df["split"] == split].reset_index(drop=True)
        ds = SkinDataset(sub, classes, img_size, train=train_flag)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=train_flag,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=num_workers > 0,
        )
        print(f"  {split}: {len(sub)} images")
    return loaders, df[df["split"] == "train"]["unified_label"].tolist(), df[df["split"] == "test"]


def class_weights(train_labels, classes, device):
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = [total / (len(classes) * counts.get(c, 1)) for c in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Standard multi-class focal loss (Lin et al. 2017), unweighted by class
    frequency -- the down-weighting of easy examples is the imbalance-handling
    mechanism being tested here, as an alternative to class-weighted CE."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = Fnn.cross_entropy(logits, target, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(loss_kind, train_labels, classes, device):
    if loss_kind == "weighted_ce":
        weights = class_weights(train_labels, classes, device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    elif loss_kind == "unweighted_ce":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif loss_kind == "focal":
        return FocalLoss(gamma=2.0, label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown --loss {loss_kind!r}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def metrics_report(y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    macro_f1 = f1.mean()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    per_class = {
        classes[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    }
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def train_one(arch, epochs, batch_size, img_size, lr, out_dir, num_workers, patience,
              exclude_classes=None, loss_kind="weighted_ce", reseed=None, grad_checkpointing=False,
              manifest_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(manifest_path) if manifest_path else ROOT / "manifest.csv"
    df_all = pd.read_csv(manifest_path)
    exclude = set(exclude_classes or [])
    classes = sorted(c for c in df_all["unified_label"].unique().tolist() if c not in exclude)
    print(f"Classes ({len(classes)}): {classes}" + (f"  [excluded: {sorted(exclude)}]" if exclude else ""))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            loaders, train_labels, test_df = build_loaders(
                manifest_path, classes, img_size, batch_size, num_workers, reseed=reseed
            )
            model = timm.create_model(arch, pretrained=True, num_classes=len(classes)).to(device)
            if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(enable=True)
                print(f"[{arch}] gradient checkpointing enabled")
            criterion = build_criterion(loss_kind, train_labels, classes, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            # ``--epochs 0`` is a genuinely uncapped early-stopping run.  A
            # validation-driven scheduler is used because cosine annealing
            # requires a predetermined horizon and would quietly reintroduce
            # a fixed cap through T_max.
            uncapped = epochs == 0
            if uncapped:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5,
                    patience=max(2, patience // 2), min_lr=1e-7,
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs
                )
            scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

            best_val_f1 = -1.0
            epochs_no_improve = 0
            history = []

            epoch = 0
            while uncapped or epoch < epochs:
                epoch += 1
                model.train()
                t0 = time.time()
                running_loss = 0.0
                n_batches = 0
                for x, y in loaders["train"]:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()
                    n_batches += 1
                y_val, pred_val = evaluate(model, loaders["val"], device)
                val_report = metrics_report(y_val, pred_val, classes)
                if uncapped:
                    scheduler.step(val_report["macro_f1"])
                else:
                    scheduler.step()
                train_loss = running_loss / max(n_batches, 1)
                dt = time.time() - t0
                epoch_target = "early-stop" if uncapped else str(epochs)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[{arch}] epoch {epoch}/{epoch_target}  train_loss={train_loss:.4f}  "
                      f"val_acc={val_report['accuracy']:.4f}  val_macroF1={val_report['macro_f1']:.4f}  "
                      f"lr={current_lr:.2e}  ({dt:.1f}s)")
                history.append({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_accuracy": val_report["accuracy"], "val_macro_f1": val_report["macro_f1"],
                    "lr": current_lr, "epoch_seconds": dt,
                })

                if val_report["macro_f1"] > best_val_f1:
                    best_val_f1 = val_report["macro_f1"]
                    epochs_no_improve = 0
                    torch.save({
                        "state_dict": model.state_dict(),
                        "arch": arch,
                        "classes": classes,
                        "img_size": img_size,
                    }, out_dir / "best.pt")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[{arch}] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                        break

            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 4:
                raise
            batch_size = max(4, batch_size // 2)
            print(f"[{arch}] CUDA OOM — retrying with batch_size={batch_size}")

    # Final test-set evaluation using the BEST checkpoint
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    y_test, pred_test = evaluate(model, loaders["test"], device)
    test_report = metrics_report(y_test, pred_test, classes)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2))

    print(f"\n=== [{arch}] TEST SET RESULTS ===")
    print(f"Accuracy: {test_report['accuracy']:.4f}   Macro-F1: {test_report['macro_f1']:.4f}")
    for c, m in test_report["per_class"].items():
        print(f"  {c:22s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  support={m['support']}")

    # Also save raw test-set logits/probs (+ filepaths + raw pre-merge labels)
    # for later analysis (ensembling, sub-label auditing, TTA comparison).
    save_test_probs(model, loaders["test"], device, classes, out_dir, test_df)


@torch.no_grad()
def save_test_probs(model, loader, device, classes, out_dir, test_df):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(y.numpy().tolist())
    all_probs = np.concatenate(all_probs, axis=0)
    np.savez(
        out_dir / "test_probs.npz",
        probs=all_probs,
        labels=np.array(all_labels),
        classes=np.array(classes),
        filepaths=np.array(test_df["filepath"].tolist()),
        raw_labels=np.array(test_df["raw_label"].tolist()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="timm model name, e.g. efficientnetv2_rw_s, convnext_tiny")
    ap.add_argument(
        "--epochs", type=int, default=18,
        help="fixed epoch count; use 0 for no epoch cap (patience-based early stopping)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=2,
                     help="default lowered from 4->2: Windows DataLoader shared-memory crashes were "
                          "observed at 4 workers under GPU/CPU contention from a second concurrent job")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--exclude-classes", default="",
                     help="comma-separated unified_label values to drop entirely from training/eval")
    ap.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "unweighted_ce", "focal"])
    ap.add_argument("--reseed", type=int, default=None,
                     help="if set, ignore manifest's stored split column and regenerate a fresh "
                          "group-stratified 70/15/15 split with this seed (for repeated-split robustness checks)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                     help="enable gradient checkpointing to reduce VRAM use (useful at larger --img-size)")
    ap.add_argument("--manifest", default=str(ROOT / "manifest.csv"),
                    help="manifest CSV to train/evaluate from")
    args = ap.parse_args()
    exclude = [c.strip() for c in args.exclude_classes.split(",") if c.strip()]
    train_one(args.arch, args.epochs, args.batch_size, args.img_size, args.lr,
              args.out, args.num_workers, args.patience,
              exclude_classes=exclude, loss_kind=args.loss, reseed=args.reseed,
              grad_checkpointing=args.grad_checkpointing, manifest_path=args.manifest)
