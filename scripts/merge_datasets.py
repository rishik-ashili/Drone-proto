#!/usr/bin/env python3
"""Merge YOLO-format datasets into one training set with our class list.

    python scripts/merge_datasets.py \\
        --add datasets/landmarks \\
        --add datasets/public/mit_helipad:helipad \\
        --add datasets/raw/20261105_101500:auto \\
        --out datasets/merged

Each --add is DIR[:MAPPING]
  (none)         class ids already follow gdnav YOLO_CLASSES
  auto           class names from DIR/classes.txt or DIR/data.yaml, matched by name
  helipad        every box becomes this one class (single-class datasets)
  0=helipad,2=pad_red   explicit id -> name mapping (unmapped ids are dropped)
DIR may contain train/val/test splits (images/<split> + labels/<split>, or <split>/images +
<split>/labels) or one flat images/ + labels/ folder (split --val-frac at random).
Files are hard-linked when possible (no extra disk), else copied.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.sim.world import YOLO_CLASSES  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_splits(d: Path) -> dict[str, tuple[Path, Path]]:
    out = {}
    for split in ("train", "val", "valid", "test"):
        for img, lab in ((d / "images" / split, d / "labels" / split), (d / split / "images", d / split / "labels")):
            if img.is_dir():
                out["val" if split == "valid" else split] = (img, lab)
                break
    if not out and (d / "images").is_dir():
        out["flat"] = (d / "images", d / "labels")
    return out


def source_names(d: Path) -> list[str] | None:
    if (d / "classes.txt").exists():
        return [l.strip() for l in (d / "classes.txt").read_text().splitlines() if l.strip()]
    if (d / "data.yaml").exists():
        names = yaml.safe_load((d / "data.yaml").read_text()).get("names")
        return [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    return None


def build_map(d: Path, spec: str | None) -> dict[int, int] | str:
    if spec is None:
        return {i: i for i in range(len(YOLO_CLASSES))}
    if spec == "auto":
        names = source_names(d)
        if not names:
            raise SystemExit(f"{d}: no classes.txt / data.yaml for 'auto'")
        return {i: YOLO_CLASSES.index(n) for i, n in enumerate(names) if n in YOLO_CLASSES}
    if spec in YOLO_CLASSES:
        return spec  # single class
    m = {}
    for part in spec.split(","):
        k, v = part.split("=")
        if v not in YOLO_CLASSES:
            raise SystemExit(f"unknown class {v}; classes are {YOLO_CLASSES}")
        m[int(k)] = YOLO_CLASSES.index(v)
    return m


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", action="append", required=True, metavar="DIR[:MAP]")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "merged"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(a.seed)
    counts: dict[str, dict[str, int]] = {}
    for item in a.add:
        dpath, _, spec = item.partition(":")
        d = Path(dpath)
        cmap = build_map(d, spec or None)
        tag = d.name
        for split, (img_dir, lab_dir) in find_splits(d).items():
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                sp = split if split != "flat" else ("val" if rng.random() < a.val_frac else "train")
                (out / "images" / sp).mkdir(parents=True, exist_ok=True)
                (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
                name = f"{tag}__{img.stem}"
                link(img, out / "images" / sp / f"{name}{img.suffix.lower()}")
                lines = []
                lab = lab_dir / f"{img.stem}.txt"
                if lab.exists():
                    for line in lab.read_text().splitlines():
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        c = int(float(parts[0]))
                        new = YOLO_CLASSES.index(cmap) if isinstance(cmap, str) else cmap.get(c)
                        if new is not None:
                            lines.append(" ".join([str(new)] + parts[1:]))
                (out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                counts.setdefault(tag, {}).setdefault(sp, 0)
                counts[tag][sp] += 1
    splits = sorted({s for c in counts.values() for s in c})
    yml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": {i: n for i, n in enumerate(YOLO_CLASSES)}}
    if "test" in splits:
        yml["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(yml, sort_keys=False))
    for tag, c in counts.items():
        print(f"  {tag:30s} " + "  ".join(f"{s}={n}" for s, n in sorted(c.items())))
    print(f"Merged dataset: {out / 'data.yaml'}\nTrain: python scripts/train_landmark_model.py --data {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
