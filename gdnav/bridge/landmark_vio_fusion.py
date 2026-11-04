"""Landmark correction fused with the VIO pose (Phase 3).

PX4's EKF2 accepts a single external-vision stream, so landmark fixes cannot be
sent separately while VIO is feeding EV. Instead the companion estimates the
transform C = (dx, dy, dyaw) that maps the drifting VIO odometry frame into the
landmark map frame:

    p_map = Rz(dyaw) * p_odom + [dx, dy]          yaw_map = yaw_odom + dyaw

Each confirmed landmark fix (and ArUco heading fix) updates C with a small
Kalman filter; the process noise grows with distance travelled, matching how
VIO drifts. The corrected pose is what gets sent to PX4. When VIO resets its
odometry frame (reset counter changes), C is re-anchored so the corrected pose
stays continuous.

Used by both the simulator (gdnav/runner.py) and the ROS 2 bridge
(jetson/ros2/gdnav_ros/vio_bridge_node.py).
"""
from __future__ import annotations

import math

import numpy as np

from gdnav.common.geometry import rot_z, wrap_pi


class LandmarkVioFusion:
    """3-DoF (dx, dy, dyaw) Kalman correction that maps the drifting VIO frame into the landmark map frame."""
    def __init__(self, drift_per_m: float = 0.02, yaw_drift_per_m: float = math.radians(0.3),
                 gate_sigma: float = 4.0, init=(0.0, 0.0, 0.0), init_pos_std: float = 0.05,
                 init_yaw_std: float = math.radians(3.0)):
        self.x = np.array(init, float)                 # dx, dy, dyaw
        # takeoff alignment between the VIO frame and the map: heading is only as good as the magnetometer
        self.P = np.diag([init_pos_std ** 2, init_pos_std ** 2, init_yaw_std ** 2])
        self.q_pos = drift_per_m
        self.q_yaw = yaw_drift_per_m
        self.gate = gate_sigma
        self.reset_counter: int | None = None
        self.last_odom: tuple[np.ndarray, float] | None = None
        self.last_corrected: tuple[np.ndarray, float] | None = None
        self.accepted = 0
        self.rejected = 0
        self.resets = 0

    # ----------------------------------------------------------------- frames
    def to_map(self, p_odom: np.ndarray, yaw_odom: float) -> tuple[np.ndarray, float]:
        """VIO-frame pose -> corrected map-frame pose."""
        dx, dy, dyaw = self.x
        p = np.asarray(p_odom, float).copy()
        p[:2] = rot_z(dyaw) @ p[:2] + np.array([dx, dy])
        return p, wrap_pi(yaw_odom + dyaw)

    def vel_to_map(self, v_odom: np.ndarray) -> np.ndarray:
        """VIO-frame velocity -> map frame (rotation only)."""
        v = np.asarray(v_odom, float).copy()
        v[:2] = rot_z(self.x[2]) @ v[:2]
        return v

    # ---------------------------------------------------------------- updates
    def on_odometry(self, p_odom: np.ndarray, yaw_odom: float, reset_counter: int = 0,
                    anchor: tuple[np.ndarray, float, float] | None = None):
        """Feed every VIO sample; returns the corrected (map-frame) pose.

        ``anchor`` = (position, yaw, std) of the flight controller's current estimate. It is
        used when VIO re-initialises its frame: the drone may have flown on optical flow
        during the outage, so the last corrected pose is stale."""
        if self.reset_counter is not None and reset_counter != self.reset_counter:
            if anchor is not None:
                pm, ym, a_std = np.asarray(anchor[0], float), float(anchor[1]), float(anchor[2])
            elif self.last_corrected is not None:
                (pm, ym), a_std = self.last_corrected, 0.05
            else:
                pm, ym, a_std = None, 0.0, 0.0
            if pm is not None:
                dyaw = wrap_pi(ym - yaw_odom)
                d = pm[:2] - rot_z(dyaw) @ np.asarray(p_odom[:2], float)
                self.x = np.array([d[0], d[1], dyaw])
                self.P = np.diag([a_std ** 2 + 1e-4, a_std ** 2 + 1e-4, math.radians(2) ** 2])
            self.resets += 1
        elif self.last_odom is not None:
            dist = float(np.linalg.norm(np.asarray(p_odom[:2]) - self.last_odom[0][:2]))
            self.P += np.diag([(self.q_pos ** 2) * dist] * 2 + [(self.q_yaw ** 2) * dist])
        self.reset_counter = reset_counter
        self.last_odom = (np.asarray(p_odom, float).copy(), float(yaw_odom))
        self.last_corrected = self.to_map(p_odom, yaw_odom)
        return self.last_corrected

    def on_landmark_fix(self, fix_xy: np.ndarray, std: float, p_odom: np.ndarray | None = None,
                        yaw_fix: float | None = None, yaw_odom: float | None = None,
                        yaw_std: float = math.radians(1.5)) -> bool:
        """Landmark-implied drone position (map frame) at the time of ``p_odom``."""
        if p_odom is None:
            if self.last_odom is None:
                return False
            p_odom, yaw_odom = self.last_odom
        po = np.asarray(p_odom[:2], float)
        dyaw = self.x[2]
        c, s = math.cos(dyaw), math.sin(dyaw)
        pred = rot_z(dyaw) @ po + self.x[:2]
        # d(Rz(dyaw) po)/d dyaw
        dR = np.array([-s * po[0] - c * po[1], c * po[0] - s * po[1]])
        H = np.array([[1.0, 0.0, dR[0]], [0.0, 1.0, dR[1]]])
        z = np.asarray(fix_xy, float)
        Rm = np.eye(2) * std * std
        if yaw_fix is not None and yaw_odom is not None:
            H = np.vstack([H, [0.0, 0.0, 1.0]])
            y = np.concatenate([z - pred, [wrap_pi(yaw_fix - (yaw_odom + dyaw))]])
            Rm = np.diag([std * std, std * std, yaw_std * yaw_std])
        else:
            y = z - pred
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > self.gate ** 2 * len(y):
            self.rejected += 1
            return False
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[2] = wrap_pi(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        self.accepted += 1
        if self.last_odom is not None:
            self.last_corrected = self.to_map(*self.last_odom)
        return True

    @property
    def correction_std(self) -> float:
        """1-sigma horizontal uncertainty of the current correction (m)."""
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))
