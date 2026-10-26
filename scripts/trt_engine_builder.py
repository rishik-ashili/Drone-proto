#!/usr/bin/env python3
"""trt_engine_builder (Phase 3): ONNX -> TensorRT engine on the Jetson, plus a
benchmark of every available backend so POC and Orin numbers can be compared.

    python3 scripts/trt_engine_builder.py --fp16                 # on the Jetson
    python3 scripts/trt_engine_builder.py --int8 --calib datasets/landmarks/images/val
    python3 scripts/trt_engine_builder.py --benchmark-only       # any machine: ORT / OpenCV timing

Re-measure accuracy after INT8 (plan, section 8): run scripts/train_landmark_model.py's
validation against the engine, not only the speed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_engine(onnx: Path, engine: Path, fp16: bool, int8: bool, calib: str | None) -> None:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        sys.exit("trtexec not found - run this on the Jetson (JetPack 6.x ships TensorRT)")
    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={engine}", "--memPoolSize=workspace:1024"]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
        if calib:
            print("note: trtexec INT8 without a calibration cache uses dummy scales; for real accuracy build the"
                  " cache with the TensorRT Python calibrator over", calib)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"engine written: {engine}")


def benchmark(onnx: Path, n: int = 100) -> None:
    import cv2

    from gdnav.perception.yolo_detector import YoloDetector

    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    det = YoloDetector(onnx)
    for _ in range(10):
        det.detect(img)
    t0 = time.perf_counter()
    for _ in range(n):
        det.detect(img)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{det.backend:32s} {ms:7.2f} ms/frame  {1000 / ms:6.1f} FPS  (imgsz {det.imgsz})")
    net = cv2.dnn.readNetFromONNX(str(onnx))
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (det.imgsz, det.imgsz)), 1 / 255.0, swapRB=True)
    for _ in range(5):
        net.setInput(blob)
        net.forward()
    t0 = time.perf_counter()
    for _ in range(n):
        net.setInput(blob)
        net.forward()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{'opencv-dnn':32s} {ms:7.2f} ms/frame  {1000 / ms:6.1f} FPS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=str(ROOT / "models/landmarks_yolov8n.onnx"))
    ap.add_argument("--engine", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--benchmark-only", action="store_true")
    a = ap.parse_args()
    onnx = Path(a.onnx)
    if not a.benchmark_only:
        engine = Path(a.engine) if a.engine else onnx.with_suffix(".int8.engine" if a.int8 else ".fp16.engine")
        build_engine(onnx, engine, a.fp16 or not a.int8, a.int8, a.calib)
    benchmark(onnx)


if __name__ == "__main__":
    main()
