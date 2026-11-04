"""Stereo visual-inertial odometry model (Phase 3: RealSense D455 / OAK-D + cuVSLAM).

This is a behavioural model, not a VIO algorithm: it produces what cuVSLAM
publishes (pose and velocity in its own odometry frame) with the error
characteristics that matter for navigation:
  * drift that grows with distance travelled (scale error + random walk)
  * slow heading drift
  * tracking loss (no output) when the stereo cameras see nothing usable, and a
    re-initialised odometry frame with a new reset counter when tracking resumes
Rates and magnitudes are in config/sim.yaml ``vio:``. Replace them with numbers
measured on the real camera before trusting comparisons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gdnav.common.geometry import rot_z, wrap_pi


@dataclass
class VioSample:
    """One VIO odometry output (pose and velocity in the VIO frame, uncertainty, reset counter)."""
    t: float
    pos: np.ndarray        # odometry-frame position (NED-aligned, metres)
    vel: np.ndarray        # odometry-frame velocity
    yaw: float             # odometry-frame heading
    pos_std: float
    yaw_std: float
    reset_counter: int
    quality: int           # 0..100 like px4_msgs/VehicleOdometry.quality


class VioSensor:
    """Behavioural stereo-VIO model: distance-scaled drift, heading drift, tracking loss, re-initialisation."""
    def __init__(self, cfg: dict, rng: np.random.Generator, start_pos=np.zeros(3), start_yaw=0.0):
        self.c = cfg
        self.rng = rng
        self.scale = 1.0 + rng.uniform(-cfg["scale_error"], cfg["scale_error"])
        self.yaw_bias_rate = math.radians(rng.normal(0, cfg["yaw_drift_deg_per_min"])) / 60.0
        self.reset_counter = 0
        self._init(np.asarray(start_pos, float), start_yaw)
        self.tracking = True
        self.degraded = False   # dim light: noisier output
        self.lost = False       # injected tracking loss
        self._buffer: list[VioSample] = []

    def _init(self, true_pos: np.ndarray, true_yaw: float) -> None:
        # a fresh odometry frame starts at the origin, aligned with the current heading error
        self.p = np.zeros(3)
        self.yaw_err = 0.0
        self.origin_true = true_pos.copy()
        self.origin_yaw = true_yaw
        self.prev_true = true_pos.copy()
        self.prev_yaw = true_yaw

    def step(self, t: float, true_pos: np.ndarray, true_vel: np.ndarray, true_yaw: float, dt: float) -> None:
        """Integrate true motion into the drifting odometry frame."""
        if self.lost:
            self.tracking = False
            self.prev_true = true_pos.copy()
            self.prev_yaw = true_yaw
            return
        if not self.tracking:  # tracking recovered: new odometry frame
            self.reset_counter += 1
            self._init(true_pos, true_yaw)
            self.tracking = True
        c = self.c
        d = true_pos - self.prev_true
        dist = float(np.linalg.norm(d))
        self.yaw_err += self.yaw_bias_rate * dt + self.rng.normal(
            0, math.radians(c["yaw_rw_deg_per_sqrt_m"]) * math.sqrt(dist + 1e-9))
        # motion is integrated in the (drifting) odometry frame relative to its origin heading
        R = rot_z(self.yaw_err - self.origin_yaw)
        step_xy = R @ d[:2] * self.scale + self.rng.normal(0, c["pos_rw_per_sqrt_m"] * math.sqrt(dist + 1e-9), 2)
        self.p[:2] += step_xy
        self.p[2] += d[2]
        self.prev_true = true_pos.copy()
        self.prev_yaw = true_yaw

    def sample(self, t: float, true_vel: np.ndarray, true_yaw: float) -> VioSample | None:
        """Produce (and buffer) the current odometry sample, or None while tracking is lost."""
        if not self.tracking:
            return None
        c = self.c
        k = 3.0 if self.degraded else 1.0
        R = rot_z(self.yaw_err - self.origin_yaw)
        vel = np.concatenate([R @ true_vel[:2] * self.scale, true_vel[2:3]])
        vel = vel + self.rng.normal(0, c["vel_noise"] * k, 3)
        pos = self.p + self.rng.normal(0, c["pos_noise"] * k, 3)
        yaw = wrap_pi(true_yaw - self.origin_yaw + self.yaw_err + self.rng.normal(0, math.radians(0.2) * k))
        s = VioSample(t, pos, vel, yaw, c["pos_noise"] * k * 2, math.radians(0.5) * k,
                      self.reset_counter, 40 if self.degraded else 95)
        self._buffer.append(s)
        return s

    def drain(self) -> list[VioSample]:
        """Samples produced since the last call."""
        out, self._buffer = self._buffer, []
        return out
