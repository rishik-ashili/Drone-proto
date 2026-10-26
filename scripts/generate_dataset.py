#!/usr/bin/env python3
"""Generate a synthetic, auto-labelled landmark dataset in YOLO format.

Every image is rendered by the same downward-camera model the simulator uses,
from randomised scenes: floor type, lighting, height (1-4.5 m), tilt, motion
blur, sensor noise, landmark size / rotation and painted distractor shapes.
Bounding boxes come from projecting the known landmark geometry, so labels are
exact and free. This is dataset v0 of the plan; real images captured at flight
height should be added to it before flying the real drone.

    python scripts/generate_dataset.py --train 1600 --val 300 --out datasets/landmarks
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.sim.camera import DownwardCamera  # noqa: E402
from gdnav.sim.world import YOLO_CLASSES, Landmark, World  # noqa: E402

LANDMARK_TYPES = ["helipad", "pad_red", "pad_blue", "cross_yellow", "aruco"]


def random_scene(rng: np.random.Generator) -> World:
    half = 4.0
    lms = []
    for _ in range(rng.integers(1, 5)):
        for _try in range(20):
            x, y = rng.uniform(-half + 1, half - 1, 2)
            if all(math.hypot(x - l.x, y - l.y) > 1.3 for l in lms):
                break
        kind = LANDMARK_TYPES[rng.integers(len(LANDMARK_TYPES))]
        size = float(rng.uniform(0.4, 1.3))
        lms.append(Landmark(name=f"lm{len(lms)}", type=kind, x=float(x), y=float(y), size=size,
                            yaw_deg=float(rng.uniform(-180, 180)),
                            id=int(rng.integers(0, 50)) if kind == "aruco" else None))
    w = World(name="scene", x_min=-half, x_max=half, y_min=-half, y_max=half,
              res=float(rng.choice([0.008, 0.01, 0.012])), landmarks=lms, obstacles=[],
              surface=str(rng.choice(["concrete", "concrete", "grass", "mat"])),
              grid_m=float(rng.choice([0.0, 1.0, 2.0])), seed=int(rng.integers(1 << 30)),
              distractors=int(rng.integers(2, 12)))
    w.texture = w.build_texture()
    # random global colour cast (white balance differences between cameras)
    cast = rng.uniform(0.85, 1.15, 3)
    w.texture = np.clip(w.texture.astype(np.float32) * cast, 0, 255).astype(np.uint8)
    return w


def landmark_outline(lm: Landmark) -> np.ndarray:
    """World-frame outline points of the painted shape (for tight boxes)."""
    h = lm.size / 2.0
    a = math.radians(lm.yaw_deg)
    if lm.type in ("helipad", "pad_red"):
        th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        pts = np.stack([h * np.cos(th), h * np.sin(th)], 1)
    elif lm.type == "cross_yellow":
        t = lm.size / 8.0  # half arm thickness (render_landmark uses size/4)
        pts = np.array([[h, -t], [h, t], [-h, -t], [-h, t], [-t, h], [t, h], [-t, -h], [t, -h]])
    else:
        pts = np.array([[h, -h], [h, h], [-h, h], [-h, -h]])  # north/east offsets
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return pts @ R.T + np.array([lm.x, lm.y])


def render_sample(rng: np.random.Generator, imgsz=(640, 480)):
    w = random_scene(rng)
    cam = DownwardCamera(w, imgsz[0], imgsz[1], float(rng.uniform(70, 90)), rng)
    target = w.landmarks[rng.integers(len(w.landmarks))]
    alt = float(rng.uniform(1.0, 4.5))
    pos = np.array([target.x + rng.uniform(-1.8, 1.8), target.y + rng.uniform(-1.8, 1.8), -alt])
    if rng.random() < 0.1:  # some frames with no landmark at all
        pos[:2] = rng.uniform(-3.5, 3.5, 2)
    tilt = math.radians(12)
    att = np.array([rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-math.pi, math.pi)])
    vel = rng.normal(0, 1.2, 3) if rng.random() < 0.4 else None
    img = cam.render(pos, att, vel, light=float(rng.uniform(0.35, 1.35)), noise=float(rng.uniform(1, 10)))
    if rng.random() < 0.15:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)  # defocus
    if rng.random() < 0.3:
        q = int(rng.integers(35, 90))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)

    labels = []
    W, H = imgsz
    for lm in w.landmarks:
        px = cam.project(pos, att, landmark_outline(lm))
        x1, y1 = px.min(0)
        x2, y2 = px.max(0)
        full = (x2 - x1) * (y2 - y1)
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        vis = (cx2 - cx1) * (cy2 - cy1) / max(full, 1e-6)
        if vis < 0.45 or min(cx2 - cx1, cy2 - cy1) < 8:
            continue
        cls = YOLO_CLASSES.index(lm.cls)
        labels.append((cls, (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H, (cx2 - cx1) / W, (cy2 - cy1) / H))
    return img, labels


def write_split(out: Path, split: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = np.zeros(len(YOLO_CLASSES), int)
    for i in range(n):
        img, labels = render_sample(rng)
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out / "labels" / split / f"{name}.txt", "w") as f:
            for c, *b in labels:
                counts[c] += 1
                f.write(f"{c} " + " ".join(f"{v:.6f}" for v in b) + "\n")
        if (i + 1) % 100 == 0:
            print(f"  {split}: {i + 1}/{n}", flush=True)
    print(f"  {split} instances per class: " + ", ".join(f"{c}={k}" for c, k in zip(YOLO_CLASSES, counts)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="datasets/landmarks")
    ap.add_argument("--train", type=int, default=1600)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    print(f"Writing dataset to {out}")
    write_split(out, "train", a.train, a.seed)
    write_split(out, "val", a.val, a.seed + 100_000)  # different scenes = 'held-out day'
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(YOLO_CLASSES)))
    print(f"Done. Dataset config: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
