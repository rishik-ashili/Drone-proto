#!/usr/bin/env python3
"""Download openly licensed landmark datasets into datasets/public/.

    python scripts/fetch_public_datasets.py --list
    python scripts/fetch_public_datasets.py mit_helipad
    python scripts/fetch_public_datasets.py kios_landing_pad        # large (~527 MB), check its licence first

Then merge and retrain (docs/ML.md):
    python scripts/merge_datasets.py --add datasets/landmarks --add datasets/public/mit_helipad:helipad
    python scripts/train_landmark_model.py --data datasets/merged/data.yaml

Licences differ per dataset: read the record before shipping a model trained on it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "public"

DATASETS = {
    "mit_helipad": {
        "what": "260 rendered images of a white 'H' helipad on varied backgrounds, YOLO labels, 1 class",
        "licence": "MIT (github.com/MRZufarrel/Helipad)",
        "map": "helipad",
        "git": ("https://github.com/MRZufarrel/Helipad.git", "2c505696088e40e423197f5019176490f82919b0",
                "datasets/helipad"),
    },
    "kios_landing_pad": {
        "what": "Aerial Landing Pad Dataset (KIOS, University of Cyprus): real + simulated drone images, YOLO labels",
        "licence": "see https://zenodo.org/records/7477560 (not verified - check before commercial use)",
        "map": "0=helipad  (their pad is not one of our classes: map it only if you fly that pad)",
        "zenodo": "7477560",
    },
}


def fetch_git(name: str, url: str, commit: str, subdir: str) -> Path:
    tmp = OUT / f".{name}_git"
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", subdir], check=True)
    subprocess.run(["git", "-C", str(tmp), "checkout", "-q", commit], check=True)
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(tmp / subdir), str(dst))
    lic = tmp / "LICENSE"
    if lic.exists():
        shutil.copy(lic, dst / "LICENSE")
    shutil.rmtree(tmp)
    for cache in dst.rglob("*.cache"):
        cache.unlink()
    return dst


def fetch_zenodo(name: str, record: str) -> Path:
    meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}"))
    print(f"  {meta['metadata']['title']}\n  licence: {meta['metadata'].get('license')}")
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in meta["files"]:
        path = dst / f["key"]
        print(f"  downloading {f['key']} ({f['size'] / 1e6:.0f} MB)")
        with urllib.request.urlopen(f["links"]["self"]) as r, open(path, "wb") as w:
            shutil.copyfileobj(r, w, 1 << 20)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                z.extractall(dst)
            path.unlink()
    (dst / "SOURCE.json").write_text(json.dumps({"zenodo": record, "license": meta["metadata"].get("license")}, indent=1))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.names:
        for k, v in DATASETS.items():
            print(f"{k}\n  {v['what']}\n  licence: {v['licence']}\n  merge with --add datasets/public/{k}:{v['map']}\n")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for n in a.names:
        d = DATASETS[n]
        print(f"Fetching {n} ...")
        dst = fetch_git(n, *d["git"]) if "git" in d else fetch_zenodo(n, d["zenodo"])
        print(f"  -> {dst}")


if __name__ == "__main__":
    sys.exit(main())
