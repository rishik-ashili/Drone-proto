"""Downward camera rendered from the floor texture with a planar homography.

For a ground point P = (x, y, 0) and camera centre C, the image point is
    p ~ K * R_cb * R_bw * (P - C)
which is a homography in (x, y, 1). Composing it with the texture-to-world
matrix gives a single 3x3 warp from the texture into the image.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from gdnav.common.geometry import R_CAM_FROM_BODY, camera_matrix, rot_from_euler
from gdnav.sim.world import World


class DownwardCamera:
    def __init__(self, world: World, width=640, height=480, hfov_deg=80.0, rng=None):
        self.world = world
        self.w, self.h = width, height
        self.K = camera_matrix(width, height, hfov_deg)
        self.rng = rng or np.random.default_rng(0)
        self._T = world.tex_from_world_matrix()
        self.border = tuple(int(v) for v in np.median(world.texture[::50, ::50].reshape(-1, 3), 0))
        self._vignette = self._make_vignette()

    def _make_vignette(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        r = np.hypot((xx - self.w / 2) / (self.w / 2), (yy - self.h / 2) / (self.h / 2))
        return (1.0 - 0.28 * np.clip(r, 0, 1.4) ** 2)[:, :, None]

    def homography(self, pos: np.ndarray, att: np.ndarray) -> np.ndarray:
        R_bw = rot_from_euler(*att).T
        M = np.array([[1.0, 0.0, -pos[0]], [0.0, 1.0, -pos[1]], [0.0, 0.0, -pos[2]]])
        return self.K @ R_CAM_FROM_BODY @ R_bw @ M

    def project(self, pos: np.ndarray, att: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Project world ground points (N, 2) into pixel coordinates (N, 2)."""
        H = self.homography(pos, att)
        pts = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
        return pts[:, :2] / pts[:, 2:3]

    def render(self, pos, att, vel=None, light=1.0, noise=3.0, blur=True) -> np.ndarray:
        H = self.homography(np.asarray(pos), np.asarray(att)) @ self._T
        img = cv2.warpPerspective(self.world.texture, H, (self.w, self.h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.border)
        # motion blur from horizontal speed over the exposure (~8 ms)
        if blur and vel is not None:
            alt = max(0.3, -float(pos[2]))
            px_per_m = self.K[0, 0] / alt
            yaw = float(att[2])
            # body-frame velocity -> image direction (x right = body y, y down = -body x)
            vb_x = math.cos(yaw) * vel[0] + math.sin(yaw) * vel[1]
            vb_y = -math.sin(yaw) * vel[0] + math.cos(yaw) * vel[1]
            du, dv = vb_y * px_per_m * 0.008, -vb_x * px_per_m * 0.008
            L = int(min(15, math.hypot(du, dv)))
            if L >= 2:
                k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
                cv2.line(k, (L - int(du / 2), L - int(dv / 2)), (L + int(du / 2), L + int(dv / 2)), 1.0, 1)
                if k.sum() > 0:
                    img = cv2.filter2D(img, -1, k / k.sum())
        f = img.astype(np.float32) * self._vignette * light
        if noise > 0:
            f += self.rng.normal(0, noise, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)
