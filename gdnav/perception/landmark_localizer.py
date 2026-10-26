"""landmark_localizer: pixel offset + altitude + camera intrinsics -> metres.

A pixel is back-projected into a ray, rotated into the world frame with the
estimated attitude, and intersected with the ground plane at the height the
rangefinder reports. The result is the landmark position relative to the
drone in local NED, from which the drone position is implied:
    drone_xy = landmark_map_xy - offset_xy
"""
from __future__ import annotations

import math

import numpy as np

from gdnav.common.geometry import R_CAM_FROM_BODY, rot_from_euler, wrap_pi


class LandmarkLocalizer:
    def __init__(self, K: np.ndarray):
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def height(self, range_m: float, att: np.ndarray) -> float:
        return max(0.05, range_m * math.cos(att[0]) * math.cos(att[1]))

    def pixel_to_offset(self, uv, att: np.ndarray, range_m: float) -> np.ndarray | None:
        """Offset (north, east) of the ground point seen at pixel uv."""
        d_c = self.Kinv @ np.array([uv[0], uv[1], 1.0])
        d_w = rot_from_euler(*att) @ (R_CAM_FROM_BODY.T @ d_c)
        if d_w[2] <= 1e-3:
            return None
        s = self.height(range_m, att) / d_w[2]
        return s * d_w[:2]

    def marker_yaw(self, corners: np.ndarray, att: np.ndarray, range_m: float,
                   marker_yaw_deg: float) -> float | None:
        """Drone heading implied by an ArUco marker's orientation.

        The marker 'up' edge (bottom-left -> top-left corner) points along the
        marker's world yaw. Projecting it with roll/pitch only (yaw = 0) gives
        its angle in the drone's heading frame; the difference is the heading.
        """
        a0 = np.array([att[0], att[1], 0.0])
        tl = self.pixel_to_offset(corners[0], a0, range_m)
        bl = self.pixel_to_offset(corners[3], a0, range_m)
        if tl is None or bl is None:
            return None
        v = tl - bl
        alpha = math.atan2(v[1], v[0])
        return wrap_pi(math.radians(marker_yaw_deg) - alpha)
