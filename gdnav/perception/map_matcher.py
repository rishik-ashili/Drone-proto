"""Floor-map matching: drift correction without landmarks (plan Phase 4).

Teach and repeat
  * teach: while the position estimate is good (recent landmark fixes), every
    downward frame is orthorectified with the estimated pose and painted into a
    georeferenced floor mosaic (FloorMap). Saved to maps/<name>.npz.
  * repeat: each frame is orthorectified at the *estimated* pose and matched
    against the mosaic around that pose (ORB features + RANSAC similarity).
    The match offset is the estimator's drift; the result is a position and
    heading fix fused exactly like a landmark fix.

The same matcher works against any georeferenced orthophoto (a stitched mapping
flight indoors, or a satellite tile outdoors), which is what the plan calls
satellite-map matching. The appearance-robust upgrade for changing scenes is a
learned descriptor (AnyLoc / DINOv2) for retrieval; the geometry stays the same.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gdnav.common.geometry import R_CAM_FROM_BODY, rot_from_euler, wrap_pi


def ground_to_image_h(K: np.ndarray, pos: np.ndarray, att: np.ndarray) -> np.ndarray:
    """Homography from ground (x, y, 1) to image pixels for a camera at pos / att."""
    R_bw = rot_from_euler(*att).T
    M = np.array([[1.0, 0.0, -pos[0]], [0.0, 1.0, -pos[1]], [0.0, 0.0, -pos[2]]])
    return K @ R_CAM_FROM_BODY @ R_bw @ M


class FloorMap:
    """Georeferenced grayscale mosaic. Pixel (col, row) <-> world (x north, y east), north up."""

    def __init__(self, x_min, x_max, y_min, y_max, res=0.01):
        self.x_min, self.x_max, self.y_min, self.y_max, self.res = x_min, x_max, y_min, y_max, res
        rows, cols = int(round((x_max - x_min) / res)), int(round((y_max - y_min) / res))
        self.acc = np.zeros((rows, cols), np.float32)
        self.wsum = np.zeros((rows, cols), np.float32)
        self._img: np.ndarray | None = None

    # ------------------------------------------------------------ geometry
    def world_to_px(self, x, y):
        return (y - self.y_min) / self.res, (self.x_max - x) / self.res

    def px_to_world(self, col, row):
        return self.x_max - row * self.res, self.y_min + col * self.res

    @property
    def image(self) -> np.ndarray:
        if self._img is None:
            img = np.where(self.wsum > 0, self.acc / np.maximum(self.wsum, 1e-6), 0)
            self._img = img.astype(np.uint8)
        return self._img

    @property
    def coverage(self) -> float:
        return float((self.wsum > 0).mean())

    # ------------------------------------------------------------- building
    def add(self, ortho: np.ndarray, x0: float, y0: float, weight: np.ndarray) -> None:
        """Paint an ortho patch whose top-left pixel is at world (x0 = north edge, y0 = west edge)."""
        c0, r0 = self.world_to_px(x0, y0)
        c0, r0 = int(round(c0)), int(round(r0))
        h, w = ortho.shape[:2]
        ra, ca = max(0, r0), max(0, c0)
        rb, cb = min(self.acc.shape[0], r0 + h), min(self.acc.shape[1], c0 + w)
        if ra >= rb or ca >= cb:
            return
        sub = ortho[ra - r0:rb - r0, ca - c0:cb - c0].astype(np.float32)
        wt = weight[ra - r0:rb - r0, ca - c0:cb - c0]
        self.acc[ra:rb, ca:cb] += sub * wt
        self.wsum[ra:rb, ca:cb] += wt
        self._img = None

    def crop(self, x_c: float, y_c: float, half_m: float) -> tuple[np.ndarray, float, float]:
        """Crop around a world point; returns (image, x_top, y_left)."""
        x_top, y_left = x_c + half_m, y_c - half_m
        c0, r0 = self.world_to_px(x_top, y_left)
        n = int(round(2 * half_m / self.res))
        c0, r0 = int(round(c0)), int(round(r0))
        out = np.zeros((n, n), np.uint8)
        img = self.image
        ra, ca, rb, cb = max(0, r0), max(0, c0), min(img.shape[0], r0 + n), min(img.shape[1], c0 + n)
        if ra < rb and ca < cb:
            out[ra - r0:rb - r0, ca - c0:cb - c0] = img[ra:rb, ca:cb]
        return out, self.x_max - r0 * self.res, self.y_min + c0 * self.res

    # ----------------------------------------------------------- save / load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, image=self.image, valid=(self.wsum > 0),
                            meta=json.dumps({"bounds": [self.x_min, self.x_max, self.y_min, self.y_max],
                                             "res": self.res}))

    @classmethod
    def load(cls, path: str | Path) -> "FloorMap":
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        m = cls(*meta["bounds"], res=meta["res"])
        m.acc = d["image"].astype(np.float32)
        m.wsum = d["valid"].astype(np.float32)
        return m

    @classmethod
    def from_image(cls, gray: np.ndarray, x_max: float, y_min: float, res: float) -> "FloorMap":
        """Wrap an existing orthophoto / satellite tile (north up)."""
        h, w = gray.shape[:2]
        m = cls(x_max - h * res, x_max, y_min, y_min + w * res, res)
        m.acc = gray.astype(np.float32)
        m.wsum = np.ones_like(m.acc)
        return m


@dataclass
class MapFix:
    xy: np.ndarray        # corrected drone position (world)
    yaw: float            # corrected heading
    std: float
    inliers: int
    shift_m: float        # size of the drift that was corrected


class MapMatcher:
    def __init__(self, K: np.ndarray, res: float = 0.01, patch_m: float = 3.0, search_m: float = 1.2,
                 min_inliers: int = 25, n_features: int = 1500):
        self.K = K
        self.res = res
        self.patch_m = patch_m      # side of the orthorectified patch around the drone
        self.search_m = search_m    # max drift searched for
        self.min_inliers = min_inliers
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # local contrast enhancement: low-texture floors (concrete) give 4x more matchable features
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_ms = 0.0
        self.stats = {"tries": 0, "fixes": 0}

    def orthorectify(self, image: np.ndarray, pos: np.ndarray, att: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Project the frame onto the ground around ``pos`` (north-up, ``res`` m/px).
        Returns (gray patch, valid mask, x_top, y_left)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        n = int(round(self.patch_m / self.res))
        x_top, y_left = pos[0] + self.patch_m / 2, pos[1] - self.patch_m / 2
        # patch pixel -> world ground point
        T = np.array([[0.0, -self.res, x_top], [self.res, 0.0, y_left], [0.0, 0.0, 1.0]])
        H = ground_to_image_h(self.K, pos, att) @ T
        h, w = gray.shape[:2]
        patch = cv2.warpPerspective(gray, H, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (n, n),
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        return self.clahe.apply(patch), valid, x_top, y_left

    def teach(self, fmap: FloorMap, image, pos, att) -> None:
        patch, valid, x_top, y_left = self.orthorectify(image, pos, att)
        # weight the centre of the frame more (less lens / attitude error)
        n = patch.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
        wt = (valid > 0) * np.clip(1.2 - r, 0.05, 1.0)
        fmap.add(patch, x_top, y_left, wt.astype(np.float32))

    def match(self, fmap: FloorMap, image, est_pos, est_att, est_yaw: float) -> MapFix | None:
        import time
        t0 = time.perf_counter()
        self.stats["tries"] += 1
        patch, valid, x_top, y_left = self.orthorectify(image, est_pos, est_att)
        half = self.patch_m / 2 + self.search_m
        ref, rx_top, ry_left = fmap.crop(est_pos[0], est_pos[1], half)
        ref_valid = (ref > 0).astype(np.uint8) * 255
        k1, d1 = self.orb.detectAndCompute(patch, valid)
        k2, d2 = self.orb.detectAndCompute(ref, cv2.erode(ref_valid, np.ones((9, 9), np.uint8)))
        self.last_ms = (time.perf_counter() - t0) * 1000
        if d1 is None or d2 is None or len(k1) < self.min_inliers or len(k2) < self.min_inliers:
            return None
        pairs = self.matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n2 in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n2.distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([k1[m.queryIdx].pt for m in good])
        dst = np.float32([k2[m.trainIdx].pt for m in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=2000, confidence=0.995)
        self.last_ms = (time.perf_counter() - t0) * 1000
        if A is None or inl is None or int(inl.sum()) < self.min_inliers:
            return None
        scale = math.hypot(A[0, 0], A[1, 0])
        if not 0.93 < scale < 1.07:
            return None
        # where does the patch centre (= estimated position) land in the reference crop?
        n = patch.shape[0]
        c = A @ np.array([n / 2, n / 2, 1.0])
        true_x = rx_top - c[1] * self.res
        true_y = ry_left + c[0] * self.res
        shift = math.hypot(true_x - est_pos[0], true_y - est_pos[1])
        if shift > self.search_m:
            return None
        # image rotation (north-up rasters): positive angle = patch content rotated clockwise in the map
        rot = math.atan2(A[1, 0], A[0, 0])
        yaw = wrap_pi(est_yaw + rot)
        ninl = int(inl.sum())
        std = 0.03 + 0.6 / math.sqrt(ninl)
        self.stats["fixes"] += 1
        return MapFix(np.array([true_x, true_y]), yaw, std, ninl, shift)
