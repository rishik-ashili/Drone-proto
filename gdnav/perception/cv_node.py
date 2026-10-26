"""cv_node: camera frame -> detections -> metric landmark observations.

Pipeline per frame
  1. detectors: ArUco (non-ML anchors) + YOLO landmark model (or colour fallback)
  2. landmark_localizer: pixel offset -> metric offset using est. attitude + range
  3. association with the landmark map (tag id, or class + nearest within a gate)
  4. temporal filter (N consecutive frames) -> confirmed observations
Confirmed observations are what the navigation logic and the estimator see.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gdnav.common.config import REPO_ROOT
from gdnav.common.geometry import camera_matrix
from gdnav.interface.vehicle import CameraFrame
from gdnav.perception.aruco_detector import ArucoDetector
from gdnav.perception.color_detector import ColorDetector
from gdnav.perception.detections import Detection, LandmarkObservation
from gdnav.perception.landmark_localizer import LandmarkLocalizer
from gdnav.perception.temporal_filter import TemporalFilter
from gdnav.sim.world import World

CLASS_COLOURS = {
    "helipad": (235, 235, 235), "pad_red": (60, 60, 255), "pad_blue": (255, 140, 40),
    "cross_yellow": (40, 220, 255), "aruco_tag": (120, 255, 120),
}


@dataclass
class PerceptionResult:
    t: float
    detections: list[Detection]
    observations: list[LandmarkObservation]
    proc_ms: float
    image_mean: float
    image_std: float
    annotated: np.ndarray | None = None
    stats: dict = field(default_factory=dict)


class CvNode:
    def __init__(self, cfg: dict, world: World, cam_cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.world = world
        self.rng = rng or np.random.default_rng(1)
        self.K = camera_matrix(cam_cfg["width"], cam_cfg["height"], cam_cfg["hfov_deg"])
        self.localizer = LandmarkLocalizer(self.K)
        self.filter = TemporalFilter(cfg["temporal_frames"])
        self.aruco = ArucoDetector() if cfg.get("use_aruco", True) else None
        self.detector, self.detector_name = self._make_detector(cfg)
        self.inject_false = False
        self.frames = 0
        self.total_dets = 0
        self._fps_t: list[float] = []

    def _make_detector(self, cfg):
        choice = cfg.get("detector", "auto")
        model = Path(cfg.get("yolo_model", ""))
        if not model.is_absolute():
            model = REPO_ROOT / model
        if choice in ("auto", "yolo") and model.exists():
            from gdnav.perception.yolo_detector import YoloDetector
            det = YoloDetector(model, imgsz=cfg["yolo_imgsz"], conf=cfg["yolo_conf"], iou=cfg["yolo_iou"])
            return det, f"YOLO ({model.name}, {det.backend})"
        if choice == "yolo":
            raise FileNotFoundError(f"YOLO model not found: {model}. Run scripts/train_landmark_model.py "
                                    "or scripts/download_models.py")
        return ColorDetector(), "Colour segmentation (no model found)"

    # ------------------------------------------------------------------
    def process(self, frame: CameraFrame, est_pos: np.ndarray, annotate: bool = True) -> PerceptionResult:
        t0 = time.perf_counter()
        img = frame.image
        dets: list[Detection] = []
        if self.aruco:
            dets += self.aruco.detect(img)
        ml = self.detector.detect(img)
        # an ArUco decode supersedes the YOLO 'aruco_tag' box on the same tag
        for d in ml:
            if d.cls == "aruco_tag" and any(_iou(d.bbox, a.bbox) > 0.3 for a in dets if a.source == "aruco"):
                continue
            dets.append(d)
        if self.inject_false and self.rng.random() < 0.5:
            dets.append(self._fake_detection(img.shape))

        obs_by_name: dict[str, LandmarkObservation] = {}
        implied: dict[str, np.ndarray] = {}
        for d in dets:
            ob = self._observe(d, frame, est_pos)
            if ob is None:
                continue
            prev = obs_by_name.get(ob.name)
            if prev is None or ob.std < prev.std:
                obs_by_name[ob.name] = ob
                implied[ob.name] = est_pos[:2] + ob.offset_ned
        confirmed = self.filter.update(implied)
        for name, ob in obs_by_name.items():
            ob.confirmed = name in confirmed

        self.frames += 1
        self.total_dets += len(dets)
        now = time.perf_counter()
        self._fps_t = [x for x in self._fps_t if now - x < 2.0] + [now]
        proc_ms = (now - t0) * 1000
        gray_small = cv2.resize(img, (80, 60))
        res = PerceptionResult(frame.t, dets, list(obs_by_name.values()), proc_ms,
                               float(gray_small.mean()), float(gray_small.std()))
        res.stats = {"fps": (len(self._fps_t) - 1) / max(1e-3, now - self._fps_t[0]) if len(self._fps_t) > 1 else 0.0,
                     "detector": self.detector_name,
                     "det_ms": getattr(self.detector, "last_ms", 0.0)}
        if annotate:
            res.annotated = self.annotate(img, dets, res)
        return res

    def _observe(self, d: Detection, frame: CameraFrame, est_pos: np.ndarray) -> LandmarkObservation | None:
        offset = self.localizer.pixel_to_offset(d.center, frame.att, frame.range_m)
        if offset is None:
            return None
        predicted = est_pos[:2] + offset
        lm = None
        if d.tag_id is not None:
            lm = next((l for l in self.world.landmarks if l.type == "aruco" and l.id == d.tag_id), None)
        else:
            cands = [l for l in self.world.landmarks if l.cls == d.cls]
            if cands:
                dist = [np.linalg.norm(l.pos - predicted) for l in cands]
                i = int(np.argmin(dist))
                if dist[i] < self.cfg["association_gate_m"]:
                    lm = cands[i]
        if lm is None:
            return None
        h = self.localizer.height(frame.range_m, frame.att)
        base = self.cfg["fix_std_aruco"] if d.source == "aruco" else self.cfg["fix_std_yolo"]
        # off-centre detections are less accurate (lens model / attitude error lever arm)
        r_px = np.hypot(d.center[0] - self.K[0, 2], d.center[1] - self.K[1, 2]) / self.K[0, 0]
        std = base + self.cfg["fix_std_per_m"] * h * (1.0 + r_px)
        yaw_fix = None
        if d.corners is not None:
            yaw_fix = self.localizer.marker_yaw(d.corners, frame.att, frame.range_m, lm.yaw_deg)
        return LandmarkObservation(lm.name, d.cls, d.source, d.conf, offset, lm.pos - offset, std, yaw_fix)

    def _fake_detection(self, shape) -> Detection:
        h, w = shape[:2]
        cls = ["pad_red", "pad_blue", "cross_yellow", "helipad"][self.rng.integers(4)]
        cx, cy = self.rng.uniform(80, w - 80), self.rng.uniform(80, h - 80)
        return Detection(cls, 0.9, (cx - 40, cy - 40, cx + 40, cy + 40), "injected")

    # ------------------------------------------------------------------
    def annotate(self, img: np.ndarray, dets: list[Detection], res: PerceptionResult) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)
        for d in dets:
            col = (0, 0, 255) if d.source == "injected" else CLASS_COLOURS.get(d.cls, (0, 255, 0))
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            if d.corners is not None:
                cv2.polylines(out, [d.corners.astype(np.int32)], True, col, 2, cv2.LINE_AA)
                cv2.circle(out, tuple(d.corners[0].astype(int)), 4, (0, 0, 255), -1)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            label = f"{d.cls}" + (f" #{d.tag_id}" if d.tag_id is not None else "") + f" {d.conf:.2f}"
            label += f" [{d.source}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cx, cy = (int(v) for v in d.center)
            cv2.line(out, (w // 2, h // 2), (cx, cy), col, 1, cv2.LINE_AA)
        for i, o in enumerate(res.observations):
            tag = "CONFIRMED" if o.confirmed else "tracking"
            txt = f"{o.name}: dN={o.offset_ned[0]:+.2f} dE={o.offset_ned[1]:+.2f} m  {tag}"
            _text(out, txt, (10, h - 12 - 20 * i), (120, 255, 120) if o.confirmed else (200, 200, 200))
        hud = f"{self.detector_name.split(' (')[0]} | {res.stats.get('fps', 0):4.1f} FPS | {res.proc_ms:5.1f} ms"
        _text(out, hud, (10, 24), (255, 255, 255), 0.55)
        return out


def _text(img, txt, org, colour, scale=0.5) -> None:
    """Text on a dark backing box (readable on any floor)."""
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    sub = img[max(0, y - th - 5):y + base + 2, max(0, x - 4):x + tw + 4]
    sub[:] = (sub * 0.35).astype(img.dtype)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
