#!/usr/bin/env python3
"""Record real (or Gazebo) camera frames for the landmark dataset, with pre-labels.

    python scripts/record_dataset.py --camera 0 --fps 3                 # USB / CSI camera
    python scripts/record_dataset.py --video flight.mp4 --fps 2         # a recorded flight video
    python scripts/record_dataset.py --gz-camera                        # Gazebo (docs/GAZEBO.md)

Writes datasets/raw/<date_time>/images/*.jpg and labels/*.txt (YOLO format) using
the current model as a pre-labeller, plus classes.txt. Import the folder into
CVAT / Label Studio / Roboflow as "YOLO 1.1", correct the boxes, export, then merge
with the synthetic set (scripts/merge_datasets.py) and retrain. Correcting boxes is
several times faster than drawing them.

Capture plan (docs/ML.md): flight height range, several lighting conditions, every
floor you will fly on, motion blur, and frames with NO landmark (negatives).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.common.config import MODELS_DIR, load_sim_config  # noqa: E402
from gdnav.sim.world import YOLO_CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", default=None)
    src.add_argument("--video", default=None)
    src.add_argument("--gz-camera", nargs="?", const="/gdnav/down_camera/image", default=None)
    ap.add_argument("--fps", type=float, default=3.0, help="frames saved per second")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--model", default=str(MODELS_DIR / "landmarks_yolov8n.onnx"))
    ap.add_argument("--conf", type=float, default=0.35, help="pre-label confidence (lower = more boxes to review)")
    ap.add_argument("--no-prelabel", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true", help="preview window (needs a display)")
    a = ap.parse_args()

    out = Path(a.out or ROOT / "datasets" / "raw" / time.strftime("%Y%m%d_%H%M%S"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(YOLO_CLASSES) + "\n")
    det = None
    if not a.no_prelabel and Path(a.model).exists():
        from gdnav.perception.yolo_detector import YoloDetector
        cfg = load_sim_config()
        det = YoloDetector(a.model, conf=a.conf, imgsz=cfg["perception"]["yolo_imgsz"])

    if a.gz_camera:
        from gdnav.interface.gz_camera import GzCamera
        gz = GzCamera(a.gz_camera)
        read = lambda: (True, gz())  # noqa: E731
    else:
        cap = cv2.VideoCapture(int(a.camera) if a.camera and a.camera.isdigit() else (a.camera or a.video))
        read = cap.read
    n, last = 0, 0.0
    video_dt = 1.0 / a.fps
    frame_i = 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) if a.video else 0
    print(f"Recording to {out}  (Ctrl+C to stop)")
    try:
        while n < a.max:
            ok, img = read()
            if not ok:
                if a.video:
                    break
                time.sleep(0.02)
                continue
            if img is None:
                time.sleep(0.02)
                continue
            frame_i += 1
            now = frame_i / src_fps if a.video and src_fps else time.monotonic()
            if now - last < video_dt:
                continue
            last = now
            name = f"frame_{n:05d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            if det is not None:
                H, W = img.shape[:2]
                for d in det.detect(img):
                    if d.cls not in YOLO_CLASSES:
                        continue
                    x1, y1, x2, y2 = d.bbox
                    lines.append(f"{YOLO_CLASSES.index(d.cls)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
            if a.show:
                cv2.imshow("record", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 50 == 0:
                print(f"  {n} frames")
    except KeyboardInterrupt:
        pass
    print(f"Saved {n} frames + pre-labels to {out}. Next: review labels in CVAT/Label Studio, then "
          f"python scripts/merge_datasets.py --add {out}")


if __name__ == "__main__":
    main()
