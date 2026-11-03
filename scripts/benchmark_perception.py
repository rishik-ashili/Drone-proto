#!/usr/bin/env python3
"""Measure the perception pipeline on the machine it will fly on (laptop, Pi, Jetson).

    python scripts/benchmark_perception.py                     # rendered arena frames
    python scripts/benchmark_perception.py --frames datasets/raw/2026-11-05   # your recorded frames
    python scripts/benchmark_perception.py --threads 4 --imgsz 320 416

Reports, per configuration, the full cv_node time (ArUco + YOLO + localizer +
association) as mean / p95 / max and the resulting FPS (plan section 10: inference
rate and end-to-end latency; P1 exit: "measured detection rate on the POC
companion recorded"). Results are appended to logs/perception_benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.common.config import LOGS_DIR, load_sim_config  # noqa: E402
from gdnav.interface.vehicle import CameraFrame  # noqa: E402
from gdnav.perception.cv_node import CvNode  # noqa: E402
from gdnav.sim.camera import DownwardCamera  # noqa: E402
from gdnav.sim.world import World  # noqa: E402


def rendered_frames(world: World, n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cam = DownwardCamera(world, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        lm = world.landmarks[i % len(world.landmarks)]
        pos = np.array([lm.x + rng.uniform(-1, 1), lm.y + rng.uniform(-1, 1), -rng.uniform(1.5, 3.0)])
        att = np.array([rng.uniform(-.05, .05), rng.uniform(-.05, .05), rng.uniform(-3, 3)])
        out.append((cam.render(pos, att), pos, att))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=None, help="folder of recorded .jpg/.png frames")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--detector", default="auto", choices=["auto", "yolo", "color"])
    a = ap.parse_args()
    cfg = load_sim_config()
    world = World.load("indoor_arena")
    if a.frames:
        files = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".jpg", ".png"))[:a.n]
        frames = [(cv2.resize(cv2.imread(str(f)), (640, 480)), np.array([0, 0, -2.0]), np.zeros(3)) for f in files]
    else:
        frames = rendered_frames(world, a.n)
    results = []
    print(f"{platform.node()} | {platform.machine()} | {len(frames)} frames\n")
    print(f"{'detector':40s} {'imgsz':>5s} {'thr':>3s} {'mean ms':>8s} {'p95 ms':>7s} {'max ms':>7s} {'FPS':>6s}")
    for imgsz in a.imgsz:
        for thr in a.threads:
            pc = dict(cfg["perception"], detector=a.detector, yolo_imgsz=imgsz)
            node = CvNode(pc, world, cfg["sensors"]["camera"])
            det = node.detector
            if hasattr(det, "sess") and det.sess is not None:  # rebuild the session with this thread count
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = thr
                det.sess = ort.InferenceSession(str(det.path), so, providers=det.sess.get_providers())
            elif thr != a.threads[0]:
                continue
            for img, pos, att in frames[:10]:  # warm-up
                node.process(CameraFrame(0, img, att, -pos[2]), pos, annotate=False)
            ms = []
            for img, pos, att in frames:
                t0 = time.perf_counter()
                node.process(CameraFrame(0, img, att, -pos[2]), pos, annotate=False)
                ms.append((time.perf_counter() - t0) * 1000)
            ms = np.array(ms)
            r = {"host": platform.node(), "machine": platform.machine(), "detector": node.detector_name,
                 "imgsz": imgsz, "threads": thr, "mean_ms": round(float(ms.mean()), 2),
                 "p95_ms": round(float(np.percentile(ms, 95)), 2), "max_ms": round(float(ms.max()), 2),
                 "fps": round(1000 / float(ms.mean()), 1), "time": time.strftime("%Y-%m-%d %H:%M")}
            results.append(r)
            print(f"{node.detector_name[:40]:40s} {imgsz:5d} {thr:3d} {r['mean_ms']:8.1f} {r['p95_ms']:7.1f} "
                  f"{r['max_ms']:7.1f} {r['fps']:6.1f}")
    LOGS_DIR.mkdir(exist_ok=True)
    f = LOGS_DIR / "perception_benchmark.json"
    old = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps(old + results, indent=1))
    print(f"\nAppended to {f}. Camera runs at {cfg['rates_hz']['camera']} Hz: mean time must stay under "
          f"{1000 / cfg['rates_hz']['camera']:.0f} ms (plan P3 target: 30 FPS = 33 ms).")


if __name__ == "__main__":
    main()
