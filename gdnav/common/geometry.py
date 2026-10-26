"""Rotation and frame helpers.

Conventions (same as PX4):
  * World frame: local NED, metres from the takeoff point. x = north, y = east,
    z = down. Altitude above ground is therefore -z.
  * Body frame: FRD (x forward, y right, z down).
  * Euler angles: roll, pitch, yaw (ZYX order), yaw measured clockwise from north.
"""
from __future__ import annotations

import math

import numpy as np


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix R_wb that maps body-frame vectors into the world frame."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


# Downward camera mounted under the airframe with the top of the image pointing
# towards the nose: image x (right) = body y, image y (down) = -body x,
# optical axis = body z (down).
R_CAM_FROM_BODY = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])
