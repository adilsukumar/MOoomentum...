"""
Data audit + taxonomy unification + leakage-safe split for SkinSense.

Walks dataset1/ (Dogs, 4 classes), datatset2/ (train/valid/test, 6 classes), and
dataset3/images (flat, label-in-filename, 6 classes). Unifies labels per PROJECT.md
decision D1, groups near-duplicate / augmented images so they can't leak across
splits (PROJECT.md decision D2), and writes a single manifest.csv used by training.

Usage:
    python scripts/prepare_data.py
"""
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# PROJECT.md decision D1: unify overlapping/duplicate-taxonomy labels into 5 classes.
LABEL_MAP = {
    "healthy": "Healthy",
    "fungal_infections": "Fungal_infection",
    "ringworm": "Fungal_infection",
    "bacterial_dermatosis": "Bacterial_dermatosis",
    "dermatitis": "Allergic_dermatitis",
    "hypersensitivity": "Allergic_dermatitis",
    "hypersensitivity_allergic_dermatosis": "Allergic_dermatitis",
    "demodicosis": "Demodicosis",
}

ROBOFLOW_RE = re.compile(r"^(?P<stem>.+?)\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
# dataset3/images filenames look like "<label>_<number>.jpg"
DATASET3_RE = re.compile(r"^(?P<label>.+)_\d+$")


def unify(raw_label: str) -> str:
    key = raw_label.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unrecognized raw label: {raw_label!r}")
    return LABEL_MAP[key]


def collect_dataset1(records):
    base = ROOT / "dataset1" / "Dogs"
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        for f in class_dir.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                records.append({"path": f, "raw_label": class_dir.name, "source": "dataset1"})


def collect_dataset2(records):
    base = ROOT / "datatset2"
    for split_dir in base.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for f in class_dir.rglob("*"):
                if f.suffix.lower() in IMG_EXTS:
                    records.append({"path": f, "raw_label": class_dir.name, "source": "datatset2"})


def collect_dataset3(records):
    base = ROOT / "dataset3" / "images"
    for f in base.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        m = DATASET3_RE.match(f.stem)
        if not m:
            print(f"  WARNING: could not parse label from {f.name}, skipping")
            continue
        records.append({"path": f, "raw_label": m.group("label"), "source": "dataset3"})


def roboflow_stem(path: Path) -> str | None:
    m = ROBOFLOW_RE.match(path.stem)
    return m.group("stem") if m else None


def phash_of(path: Path) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return str(imagehash.phash(im))
    except Exception as e:
        print(f"  WARNING: could not hash {path}: {e}")
        return None


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    records = []
    print("Collecting dataset1 (Dogs, 4 classes)...")
    collect_dataset1(records)
    print("Collecting datatset2 (train/valid/test, 6 classes)...")
    collect_dataset2(records)
    print("Collecting dataset3 (flat pool, 6 classes via filename)...")
    collect_dataset3(records)
    print(f"Total raw records: {len(records)}")

    # Unify labels
    for r in records:
        r["unified_label"] = unify(r["raw_label"])

    # --- Grouping for leakage-safe splitting ---
    # Group 1: exact-duplicate-file grouping via md5 (catches literal copies across datasets)
    # Group 2: Roboflow-augmentation family grouping via shared stem before ".rf.<hash>"
    # Group 3: near-duplicate grouping via perceptual hash (catches recompressed/resized dupes)
    uf = UnionFind()
    md5_of_idx = {}
    rf_stem_of_idx = {}
    phash_of_idx = {}

    print("Hashing all images (md5 + roboflow-stem + phash)... this takes a few minutes.")
    for i, r in enumerate(records):
        p = r["path"]
        data = p.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        md5_of_idx[i] = md5

        stem = roboflow_stem(p)
        rf_stem_of_idx[i] = stem

        ph = phash_of(p)
        phash_of_idx[i] = ph

        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(records)}")

    md5_buckets = defaultdict(list)
    rf_buckets = defaultdict(list)
    phash_buckets = defaultdict(list)
    for i in range(len(records)):
        md5_buckets[md5_of_idx[i]].append(i)
        if rf_stem_of_idx[i]:
            rf_buckets[(records[i]["source"], Path(records[i]["path"]).parent, rf_stem_of_idx[i])].append(i)
        if phash_of_idx[i]:
            phash_buckets[phash_of_idx[i]].append(i)

    for bucket in list(md5_buckets.values()) + list(rf_buckets.values()) + list(phash_buckets.values()):
        for j in bucket[1:]:
            uf.union(bucket[0], j)

    group_id_of_idx = {i: uf.find(i) for i in range(len(records))}
    # renumber groups to small ints for readability
    unique_groups = sorted(set(group_id_of_idx.values()))
    group_renumber = {g: n for n, g in enumerate(unique_groups)}
    for i, r in enumerate(records):
        r["group_id"] = group_renumber[group_id_of_idx[i]]

    n_groups = len(unique_groups)
    print(f"Collapsed {len(records)} images into {n_groups} leakage-safe groups "
          f"({len(records) - n_groups} images were duplicates/augmentation-siblings of another image).")

    # --- Group-level stratified split (70/15/15) ---
    import random
    random.seed(42)
    group_to_indices = defaultdict(list)
    for i, r in enumerate(records):
        group_to_indices[r["group_id"]].append(i)

    # majority unified_label per group decides its stratum
    group_label = {}
    for g, idxs in group_to_indices.items():
        labels = [records[i]["unified_label"] for i in idxs]
        group_label[g] = max(set(labels), key=labels.count)

    labels_to_groups = defaultdict(list)
    for g, lbl in group_label.items():
        labels_to_groups[lbl].append(g)

    split_of_group = {}
    for lbl, groups in labels_to_groups.items():
        groups = groups[:]
        random.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for g in groups[:n_train]:
            split_of_group[g] = "train"
        for g in groups[n_train:n_train + n_val]:
            split_of_group[g] = "val"
        for g in groups[n_train + n_val:]:
            split_of_group[g] = "test"

    for r in records:
        r["split"] = split_of_group[r["group_id"]]

    # --- Write manifest ---
    out_path = ROOT / "manifest.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "raw_label", "unified_label", "source", "group_id", "split"])
        for r in records:
            writer.writerow([str(r["path"]), r["raw_label"], r["unified_label"], r["source"], r["group_id"], r["split"]])
    print(f"Wrote manifest: {out_path} ({len(records)} rows)")

    # --- Summary report ---
    print("\n=== Per unified-class counts by split ===")
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["unified_label"]][r["split"]] += 1
    for lbl in sorted(counts):
        c = counts[lbl]
        total = sum(c.values())
        print(f"  {lbl:22s} train={c['train']:5d}  val={c['val']:5d}  test={c['test']:5d}  total={total:5d}")

    print("\n=== Raw label -> unified label mapping used ===")
    for k, v in LABEL_MAP.items():
        print(f"  {k:40s} -> {v}")


if __name__ == "__main__":
    main()
