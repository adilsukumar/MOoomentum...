"""
Audit the dedup groups produced by scripts/prepare_data.py: for every group with
more than one member, record WHICH criterion (exact md5, Roboflow-stem family,
or perceptual hash) joined its members, so the riskiest kind of merge (phash --
the only fuzzy/collision-prone one) can be sampled for manual visual review.

Does not change manifest.csv or re-run training; read-only audit.

Usage:
    python scripts/dedup_audit.py
Writes:
    models/dedup_audit_groups.json  -- full group->method->members breakdown
    models/dedup_audit_sample.json  -- ~50 sampled candidate image paths for
                                        manual visual review, weighted toward
                                        phash-only groups and any group touching
                                        Bacterial_dermatosis
"""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW_RE = re.compile(r"^(?P<stem>.+?)\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)


def roboflow_stem(path: Path):
    m = ROBOFLOW_RE.match(path.stem)
    return m.group("stem") if m else None


def main():
    manifest = pd.read_csv(ROOT / "manifest.csv")
    print(f"Re-hashing {len(manifest)} images to classify each group's merge method "
          f"(this repeats prepare_data.py's hashing pass, read-only)...")

    md5_of = {}
    rf_of = {}
    phash_of = {}
    for i, row in manifest.iterrows():
        p = Path(row["filepath"])
        data = p.read_bytes()
        md5_of[i] = hashlib.md5(data).hexdigest()
        rf_of[i] = roboflow_stem(p)
        try:
            with Image.open(p) as im:
                phash_of[i] = str(imagehash.phash(im.convert("RGB")))
        except Exception:
            phash_of[i] = None
        if (i + 1) % 1500 == 0:
            print(f"  {i+1}/{len(manifest)}")

    # Group manifest rows by their already-assigned group_id (from prepare_data.py)
    groups = defaultdict(list)
    for i, row in manifest.iterrows():
        groups[row["group_id"]].append(i)

    multi_groups = {g: idxs for g, idxs in groups.items() if len(idxs) > 1}
    print(f"\n{len(multi_groups)} groups have more than one member "
          f"({sum(len(v) for v in multi_groups.values())} images total).")

    report = {}
    method_counts = defaultdict(int)
    bacterial_multi_groups = []

    for g, idxs in multi_groups.items():
        md5s = set(md5_of[i] for i in idxs)
        rfs = set((manifest.at[i, "source"], Path(manifest.at[i, "filepath"]).parent.as_posix(), rf_of[i])
                  for i in idxs if rf_of[i])
        phs = set(phash_of[i] for i in idxs if phash_of[i])

        methods = []
        if len(md5s) < len(idxs):
            methods.append("md5_exact")
        if rfs and any(rf_of[i] for i in idxs):
            # crude check: at least one pair shares a roboflow stem+source+dir
            stem_counts = defaultdict(int)
            for i in idxs:
                if rf_of[i]:
                    stem_counts[(manifest.at[i, "source"], Path(manifest.at[i, "filepath"]).parent.as_posix(), rf_of[i])] += 1
            if any(c > 1 for c in stem_counts.values()):
                methods.append("roboflow_stem")
        if len(phs) < len([i for i in idxs if phash_of[i]]):
            methods.append("phash")
        if not methods:
            methods.append("unknown/transitive")  # union-find chain across multiple criteria

        labels = set(manifest.at[i, "unified_label"] for i in idxs)
        raw_labels = set(manifest.at[i, "raw_label"] for i in idxs)
        paths = [manifest.at[i, "filepath"] for i in idxs]

        report[str(g)] = {
            "methods": methods,
            "n_members": len(idxs),
            "unified_labels": sorted(labels),
            "raw_labels": sorted(raw_labels),
            "cross_label": len(labels) > 1,
            "paths": paths,
        }
        for m in methods:
            method_counts[m] += 1
        if "Bacterial_dermatosis" in labels:
            bacterial_multi_groups.append(str(g))

    print("\n=== Groups by contributing method (a group can have >1 method) ===")
    for m, c in sorted(method_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:20s} {c} groups")

    cross_label_groups = {g: r for g, r in report.items() if r["cross_label"]}
    print(f"\n{len(cross_label_groups)} groups contain MORE THAN ONE unified label "
          f"(same photo/near-duplicate labeled differently across source datasets) -- these are the ones "
          f"most worth checking by eye.")

    print(f"\n{len(bacterial_multi_groups)} multi-member groups touch Bacterial_dermatosis.")
    for g in bacterial_multi_groups:
        r = report[g]
        print(f"  group {g}: methods={r['methods']} raw_labels={r['raw_labels']} paths={r['paths']}")

    (ROOT / "models" / "dedup_audit_groups.json").write_text(json.dumps(report, indent=2))

    # ---- Build a ~50-image manual-review sample, weighted toward risk ----
    random.seed(7)
    phash_only_groups = [g for g, r in report.items() if r["methods"] == ["phash"]]
    sample_groups = []
    sample_groups += bacterial_multi_groups  # always include every Bacterial-touching group
    sample_groups += list(cross_label_groups.keys())  # always include every cross-label group
    remaining_slots = max(0, 20 - len(sample_groups))
    pool = [g for g in phash_only_groups if g not in sample_groups]
    random.shuffle(pool)
    sample_groups += pool[:remaining_slots]

    sample = {g: report[g] for g in dict.fromkeys(sample_groups)}  # dedupe, preserve order
    (ROOT / "models" / "dedup_audit_sample.json").write_text(json.dumps(sample, indent=2))
    print(f"\nWrote models/dedup_audit_groups.json (all {len(report)} multi-member groups)")
    print(f"Wrote models/dedup_audit_sample.json ({len(sample)} groups selected for manual visual review: "
          f"all Bacterial-touching groups, all cross-label groups, plus a random sample of phash-only groups)")


if __name__ == "__main__":
    main()
