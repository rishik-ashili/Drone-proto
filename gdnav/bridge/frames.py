"""Frame conversions for the VIO -> PX4 bridge (Phase 3).

ROS uses ENU world / FLU body; PX4 uses NED world / FRD body. Getting this
wrong is the classic cause of 'toilet-bowling' in position hold (plan,
section 11), so it lives in one tested module that both the ROS 2 bridge node
and the unit tests use.

Quaternions are [w, x, y, z] (Hamilton), rotating body -> world.
"""
from __future__ import annotations

import math

import numpy as np

_S = math.sqrt(0.5)
# rotation taking ENU world vectors into NED world: 180 deg about (1, 1, 0)/sqrt(2)
Q_NED_ENU = np.array([0.0, _S, _S, 0.0])
# rotation taking FRD body vectors into FLU body: 180 deg about x
Q_FLU_FRD = np.array([0.0, 1.0, 0.0, 0.0])


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.concatenate([[0.0], v])
    qc = q * np.array([1, -1, -1, -1])
    return q_mul(q_mul(q, qv), qc)[1:]


def enu_to_ned(v) -> np.ndarray:
    x, y, z = v
    return np.array([y, x, -z])


def flu_to_frd(v) -> np.ndarray:
    x, y, z = v
    return np.array([x, -y, -z])


def q_enu_flu_to_ned_frd(q_enu_flu: np.ndarray) -> np.ndarray:
    """Attitude of a FLU body in ENU -> attitude of the FRD body in NED."""
    q = q_mul(q_mul(Q_NED_ENU, np.asarray(q_enu_flu, float)), Q_FLU_FRD)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def yaw_from_q(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def q_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def ros_odometry_to_px4(pos_enu, q_enu_flu, vel_flu=None):
    """Convert a ROS odometry sample to PX4 VehicleOdometry fields
    (position NED, quaternion NED/FRD, velocity in FRD body frame)."""
    pos = enu_to_ned(pos_enu)
    q = q_enu_flu_to_ned_frd(q_enu_flu)
    vel = flu_to_frd(vel_flu) if vel_flu is not None else np.full(3, np.nan)
    return pos, q, vel
