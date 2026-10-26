#!/usr/bin/env python3
"""Download open-source pretrained models.

  * yolov8n.pt  - Ultralytics YOLOv8 nano, COCO-pretrained (starting point for
                  training the landmark model, and the "pretrained YOLO nano"
                  of Phase 1)
  * yolov8n_coco.onnx - the same network exported to ONNX (needs ultralytics),
                  usable with --detector yolo --yolo-model models/yolov8n_coco.onnx
                  to detect everyday COCO objects as stand-in landmarks

The custom landmark model (models/landmarks_yolov8n.onnx) ships with the
repository; re-create it with scripts/train_landmark_model.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
}


def download(name: str) -> Path:
    MODELS.mkdir(exist_ok=True)
    dst = MODELS / name
    if dst.exists():
        print(f"{dst} already present")
        return dst
    print(f"Downloading {URLS[name]} ...")
    with urllib.request.urlopen(URLS[name]) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-coco-onnx", action="store_true", help="also export yolov8n_coco.onnx")
    a = ap.parse_args()
    pt = download("yolov8n.pt")
    if a.export_coco_onnx:
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics needed for ONNX export: pip install -r requirements-train.txt")
        onnx = Path(YOLO(str(pt)).export(format="onnx", imgsz=640, opset=12))
        shutil.move(str(onnx), MODELS / "yolov8n_coco.onnx")
        print(f"  -> {MODELS / 'yolov8n_coco.onnx'}")


if __name__ == "__main__":
    main()
