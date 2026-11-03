#!/usr/bin/env python3
"""Downward-camera intrinsic calibration with a ChArUco board (OpenCV).

  1. print the board:      python scripts/calibrate_camera.py --make-board        (printables/charuco_A4.pdf)
  2. capture 20-40 views:  python scripts/calibrate_camera.py --capture 0 --out datasets/calib
                           (SPACE = save frame, Q = quit; tilt the board, cover the corners of the image)
  3. calibrate:            python scripts/calibrate_camera.py --images datasets/calib
     -> config/camera_calibration.yaml, used automatically by the perception pipeline
        (set sensors.camera.calibration: config/camera_calibration.yaml in your config)

The board uses DICT_5X5_100 so it is never confused with the landmark tags (DICT_4X4_50).
Camera-IMU extrinsics for VIO (Phase 3) are a separate step with Kalibr.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SQUARES = (5, 7)            # columns, rows
SQUARE_M = 0.035
MARKER_M = 0.026
DICT = cv2.aruco.DICT_5X5_100


def board():
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, cv2.aruco.getPredefinedDictionary(DICT))


def make_board(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w_mm, h_mm = SQUARES[0] * SQUARE_M * 1000, SQUARES[1] * SQUARE_M * 1000
    img = board().generateImage((int(w_mm / 25.4 * 300), int(h_mm / 25.4 * 300)), marginSize=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    ax = fig.add_axes([(210 - w_mm) / 2 / 210, (297 - h_mm) / 2 / 297, w_mm / 210, h_mm / 297])
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    fig.text(0.5, 0.03, f"ChArUco {SQUARES[0]}x{SQUARES[1]}, square {SQUARE_M * 1000:.0f} mm, marker {MARKER_M * 1000:.0f} mm, "
                        "DICT_5X5_100. Print at 100 %, check one square = 35 mm, glue flat.", ha="center", fontsize=8)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    cv2.imwrite(str(out.with_suffix(".png")), img)
    return out


def detect(gray: np.ndarray):
    det = cv2.aruco.CharucoDetector(board())
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = board().matchImagePoints(corners, ids)
    return obj, img


def calibrate(images: list[np.ndarray]) -> dict:
    objs, imgs, size = [], [], None
    for im in images:
        gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        r = detect(gray)
        if r is not None:
            objs.append(r[0])
            imgs.append(r[1])
    if len(objs) < 8:
        raise RuntimeError(f"only {len(objs)} usable views (need 8+); capture more, tilted, covering the image corners")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    hfov = 2 * math.degrees(math.atan(size[0] / 2 / K[0, 0]))
    return {"width": int(size[0]), "height": int(size[1]), "K": K.tolist(), "dist": dist.ravel().tolist(),
            "rms_px": round(float(rms), 4), "hfov_deg": round(hfov, 2), "views": len(objs)}


def capture(source: str, out: Path) -> None:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.png")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        vis = frame.copy()
        if r is not None:
            for p in r[1].reshape(-1, 2):
                cv2.circle(vis, tuple(int(v) for v in p), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"saved {n}  SPACE=save Q=quit  corners: {0 if r is None else len(r[1])}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibration", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and r is not None:
            cv2.imwrite(str(out / f"calib_{n:03d}.png"), frame)
            n += 1
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-board", action="store_true")
    ap.add_argument("--capture", default=None, metavar="CAMERA")
    ap.add_argument("--images", default=None, metavar="DIR")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.make_board:
        print("Board:", make_board(Path(a.out or ROOT / "printables" / "charuco_A4.pdf")))
    elif a.capture is not None:
        capture(a.capture, Path(a.out or ROOT / "datasets" / "calib"))
    elif a.images:
        files = sorted(glob.glob(str(Path(a.images) / "*.png")) + glob.glob(str(Path(a.images) / "*.jpg")))
        res = calibrate([cv2.imread(f) for f in files])
        out = Path(a.out or ROOT / "config" / "camera_calibration.yaml")
        out.write_text(yaml.safe_dump(res, sort_keys=False))
        print(yaml.safe_dump(res, sort_keys=False))
        print(f"Written {out}. RMS reprojection error {res['rms_px']} px (good: < 0.5 px).")
        print("Enable it: sensors.camera.calibration: config/camera_calibration.yaml")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
