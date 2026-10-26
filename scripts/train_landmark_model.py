#!/usr/bin/env python3
"""Train the custom landmark detector (YOLOv8n) and export it to ONNX.

Steps
  1. starts from the open-source COCO-pretrained yolov8n.pt (downloaded
     automatically, ~6 MB) - transfer learning converges in minutes
  2. trains on the synthetic dataset from scripts/generate_dataset.py
  3. evaluates precision / recall / mAP on the held-out split
  4. exports models/landmarks_yolov8n.onnx (+ .json with class names and the
     evaluation report) which the simulator, the Raspberry Pi and the Jetson
     (via scripts/trt_engine_builder.py) all load

    python scripts/train_landmark_model.py --epochs 40            # CPU ok
    python scripts/train_landmark_model.py --epochs 80 --device 0 # NVIDIA GPU

Licence note: Ultralytics is AGPL-3.0. See docs/ML.md for the alternatives the
plan lists (YOLOX-nano, Apache-2.0) if the code must stay proprietary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.sim.world import YOLO_CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "datasets/landmarks/data.yaml"))
    ap.add_argument("--weights", default="yolov8n.pt", help="starting checkpoint (COCO pretrained)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "models/landmarks_yolov8n.onnx"))
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed: pip install -r requirements-train.txt")
    if not Path(a.data).exists():
        sys.exit(f"{a.data} not found - run scripts/generate_dataset.py first")

    weights = a.weights
    local = ROOT / "models" / a.weights
    if local.exists():
        weights = str(local)
    t0 = time.time()
    model = YOLO(weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                workers=a.workers, project=str(ROOT / "runs"), name="landmarks", exist_ok=True,
                patience=15, cos_lr=True, close_mosaic=5, degrees=180, flipud=0.0, fliplr=0.0,
                hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, scale=0.4, mosaic=1.0, plots=True, verbose=False)
    train_min = (time.time() - t0) / 60
    best = ROOT / "runs/landmarks/weights/best.pt"
    model = YOLO(str(best))

    metrics = model.val(data=a.data, imgsz=a.imgsz, device=a.device, split="val", plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    report = {
        "model": "yolov8n",
        "classes": names,
        "imgsz": a.imgsz,
        "epochs": a.epochs,
        "train_minutes": round(train_min, 1),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "per_class_mAP50_95": {names[i]: round(float(v), 4) for i, v in enumerate(metrics.box.maps)},
    }
    print(json.dumps(report, indent=2))

    onnx_path = Path(model.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, out)
    shutil.copy(best, out.with_suffix(".pt"))
    out.with_suffix(".json").write_text(json.dumps({"names": names, "eval": report}, indent=2))
    assert names == YOLO_CLASSES, f"class order mismatch {names} vs {YOLO_CLASSES}"
    print(f"\nExported {out} ({out.stat().st_size / 1e6:.1f} MB) - the simulator now uses it automatically.")


if __name__ == "__main__":
    main()
