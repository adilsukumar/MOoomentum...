"""Create deterministic contact sheets for manual review of 50 dedup groups."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
GROUPS = json.loads((ROOT / "models/dedup_audit_groups.json").read_text())
OUT = ROOT / "models/dedup_manual_review"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(20260823)


def bucket(item):
    methods = set(item[1]["methods"])
    if methods == {"phash"}:
        return "phash_only"
    if "roboflow_stem" in methods:
        return "roboflow"
    if "md5_exact" in methods:
        return "md5"
    return "other"


by_bucket = {name: [] for name in ("phash_only", "roboflow", "md5")}
for item in GROUPS.items():
    name = bucket(item)
    if name in by_bucket:
        by_bucket[name].append(item)

# Perceptual-only is the method most capable of false-positive merging, so it
# receives the largest share. Include every bacterial-touching group possible.
targets = {"phash_only": 20, "roboflow": 15, "md5": 15}
sample = []
used = set()
for name, n in targets.items():
    population = [x for x in by_bucket[name] if x[0] not in used]
    bacterial = [x for x in population if "Bacterial_dermatosis" in x[1]["unified_labels"]]
    chosen = bacterial[:n]
    remaining = [x for x in population if x not in chosen]
    chosen += random.sample(remaining, min(n - len(chosen), len(remaining)))
    sample.extend((name, gid, g) for gid, g in chosen)
    used.update(gid for gid, _ in chosen)

# Some corpora have no phash-only group because every exact duplicate also,
# correctly, collides perceptually. Fill any shortfall from as-yet-unseen
# groups carrying phash evidence, then from any unseen duplicate group.
if len(sample) < 50:
    phash_candidates = [x for x in GROUPS.items()
                        if x[0] not in used and "phash" in x[1]["methods"]]
    random.shuffle(phash_candidates)
    for gid, g in phash_candidates[:50-len(sample)]:
        sample.append(("phash_overlap", gid, g)); used.add(gid)
if len(sample) < 50:
    remainder = [x for x in GROUPS.items() if x[0] not in used]
    random.shuffle(remainder)
    for gid, g in remainder[:50-len(sample)]:
        sample.append(("other_overlap", gid, g)); used.add(gid)

review_manifest = []
thumb = 220
row_h = 285
for sheet_idx in range(5):
    rows = sample[sheet_idx * 10:(sheet_idx + 1) * 10]
    canvas = Image.new("RGB", (thumb * 2 + 40, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, (method, gid, group) in enumerate(rows):
        paths = [Path(p) for p in group["paths"][:2]]
        y = row_idx * row_h
        for col, path in enumerate(paths):
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    x = 10 + col * (thumb + 20) + (thumb - im.width) // 2
                    canvas.paste(im, (x, y + 40 + (thumb - im.height) // 2))
            except Exception:
                draw.rectangle((10 + col * (thumb + 20), y + 40,
                                10 + col * (thumb + 20) + thumb, y + 40 + thumb),
                               outline="red", width=3)
        title = f"#{sheet_idx*10+row_idx+1:02d} group={gid} method={method} labels={','.join(group['raw_labels'])}"
        draw.text((8, y + 7), title[:78], fill="black")
        review_manifest.append({
            "review_index": sheet_idx * 10 + row_idx + 1,
            "group_id": gid,
            "method_bucket": method,
            "methods": group["methods"],
            "raw_labels": group["raw_labels"],
            "paths_shown": [str(p) for p in paths],
            "touches_bacterial": "Bacterial_dermatosis" in group["unified_labels"],
        })
    canvas.save(OUT / f"sheet_{sheet_idx+1}.jpg", quality=92)

(OUT / "review_manifest.json").write_text(json.dumps(review_manifest, indent=2))
print({k: len(v) for k, v in by_bucket.items()})
print(f"wrote {len(review_manifest)} review rows to {OUT}")
