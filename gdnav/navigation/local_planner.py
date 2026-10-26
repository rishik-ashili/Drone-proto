"""local_planner: reactive obstacle avoidance from a forward depth scan.

A light-weight stand-in for EGO-Planner (Phase 4): obstacle points from the
depth sensor push the commanded velocity away (repulsive field), and a
tangential component makes the drone slide around the obstacle on the side
closest to the goal instead of stopping in front of it. The interface
(desired velocity in, safe velocity out) is the same one an EGO-Planner
wrapper would expose.
"""
from __future__ import annotations

import math

import numpy as np


class LocalPlanner:
    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", True)
        self.influence = cfg.get("influence_m", 2.2)
        self.gain = cfg.get("gain", 1.8)
        self.active = False
        self.closest = float("inf")

    def adjust(self, v_des: np.ndarray, pos_xy: np.ndarray, yaw: float, scan) -> np.ndarray:
        self.active = False
        self.closest = float("inf")
        if not self.enabled or scan is None:
            return v_des
        angles, ranges = scan
        rep = np.zeros(2)
        for a, r in zip(angles, ranges):
            if r >= self.influence or r <= 0.05:
                continue
            th = yaw + a
            away = -np.array([math.cos(th), math.sin(th)])
            rep += self.gain * (1.0 / r - 1.0 / self.influence) * away
            self.closest = min(self.closest, float(r))
        n = np.linalg.norm(rep)
        speed = float(np.linalg.norm(v_des))
        if n < 1e-6 or speed < 1e-3:
            return v_des
        self.active = True
        tang = np.array([-rep[1], rep[0]])
        if np.dot(tang, v_des) < 0:
            tang = -tang
        v = v_des + rep * speed / max(n, 1.0) + tang / n * min(n, 1.0) * speed
        # slow down close to the obstacle
        limit = max(0.4, min(speed, 0.8 * (self.closest - 0.4)))
        vn = np.linalg.norm(v)
        return v / vn * limit if vn > 1e-6 else v_des
