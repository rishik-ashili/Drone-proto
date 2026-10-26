"""EKF2 analogue: GPS-free local position estimator.

Runs "on the flight controller" and fuses:
  * IMU (prediction, 200 Hz)
  * optical flow velocity, scaled by the rangefinder (EKF2_OF_*)
  * rangefinder + barometer for height (EKF2_RNG_*, EKF2_BARO_*)
  * magnetometer heading
  * external vision position / yaw (EKF2_EV_CTRL) - in this project these are
    the landmark fixes sent by the companion computer, exactly as VIO or
    AprilTag poses would be sent to a real Pixhawk over MAVLink / uXRCE-DDS.

No GNSS is used anywhere. Every external update passes a chi-square
innovation gate so a false YOLO detection cannot drag the estimate away.
"""
from __future__ import annotations

import math

import numpy as np

from gdnav.common.geometry import rot_from_euler, rot_z, wrap_pi

G = 9.81


class PositionEstimator:
    def __init__(self, cfg: dict, use_external_vision: bool = True, yaw0: float = 0.0):
        self.cfg = cfg
        self.use_ev = use_external_vision
        # horizontal: [px, py, vx, vy, bax, bay] (accel bias in body frame)
        self.xh = np.zeros(6)
        self.Ph = np.diag([0.01, 0.01, 0.01, 0.01, 0.05, 0.05])
        # vertical: [pz, vz, baz]
        self.xv = np.zeros(3)
        self.Pv = np.diag([0.01, 0.01, 0.05])
        # heading: [yaw, gyro_bias_z]
        self.xy = np.array([yaw0, 0.0])
        self.Py = np.diag([math.radians(3) ** 2, 0.003 ** 2])
        self.roll = 0.0
        self.pitch = 0.0
        self.ev_accepted = 0
        self.ev_rejected = 0
        self.flow_rejected = 0
        self.last_flow_t = -1.0
        self.last_ev_t = -1.0

    # -------------------------------------------------------------- properties
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.xh[0], self.xh[1], self.xv[0]])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.xh[2], self.xh[3], self.xv[1]])

    @property
    def yaw(self) -> float:
        return float(self.xy[0])

    @property
    def att(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    @property
    def pos_std(self) -> float:
        """Horizontal 1-sigma position uncertainty (what PX4 reports as eph)."""
        return float(math.sqrt(max(self.Ph[0, 0], self.Ph[1, 1])))

    def reset_on_ground(self) -> None:
        self.xh[2:4] = 0.0
        self.xv[1] = 0.0

    # --------------------------------------------------------------- predict
    def predict(self, accel_b: np.ndarray, gyro_b: np.ndarray, roll: float, pitch: float, dt: float) -> None:
        self.roll, self.pitch = roll, pitch
        # heading
        self.xy[0] = wrap_pi(self.xy[0] + (gyro_b[2] - self.xy[1]) * dt)
        Fy = np.array([[1.0, -dt], [0.0, 1.0]])
        Qy = np.diag([(0.003 * dt) ** 2 + 1e-9, (1e-5) ** 2 * dt])
        self.Py = Fy @ self.Py @ Fy.T + Qy

        R = rot_from_euler(roll, pitch, self.yaw)
        a_w = R @ accel_b + np.array([0.0, 0.0, G])
        Rz = rot_z(self.yaw)
        b = self.xh[4:6]
        a_h = a_w[:2] - Rz @ b
        self.xh[0:2] += self.xh[2:4] * dt + 0.5 * a_h * dt * dt
        self.xh[2:4] += a_h * dt
        F = np.eye(6)
        F[0:2, 2:4] = np.eye(2) * dt
        F[2:4, 4:6] = -Rz * dt
        q = self.cfg["accel_process_noise"]
        qb = self.cfg["bias_process_noise"]
        Q = np.diag([0.25 * q * q * dt ** 4] * 2 + [q * q * dt * dt] * 2 + [qb * qb * dt] * 2)
        self.Ph = F @ self.Ph @ F.T + Q

        az = a_w[2] - self.xv[2]
        self.xv[0] += self.xv[1] * dt + 0.5 * az * dt * dt
        self.xv[1] += az * dt
        Fv = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
        Qv = np.diag([0.25 * q * q * dt ** 4, q * q * dt * dt, qb * qb * dt])
        self.Pv = Fv @ self.Pv @ Fv.T + Qv

    # --------------------------------------------------------------- updates
    @staticmethod
    def _kf_update(x, P, z, H, Rm, gate_sigma=None):
        y = z - H @ x
        S = H @ P @ H.T + Rm
        Si = np.linalg.inv(S)
        if gate_sigma is not None:
            d2 = float(y @ Si @ y)
            if d2 > gate_sigma ** 2 * len(z):
                return False
        K = P @ H.T @ Si
        x += K @ y
        P[:] = (np.eye(len(x)) - K @ H) @ P
        return True

    def update_flow(self, t: float, vel_body: np.ndarray, std: float) -> bool:
        Rz = rot_z(self.yaw)
        # measurement model: v_body = Rz^T v_world
        H = np.zeros((2, 6))
        H[:, 2:4] = Rz.T
        ok = self._kf_update(self.xh, self.Ph, vel_body, H, np.eye(2) * std * std,
                             self.cfg["flow_gate_sigma"])
        if ok:
            self.last_flow_t = t
        else:
            self.flow_rejected += 1
        return ok

    def update_range(self, rng: float, std: float = 0.03) -> None:
        h = rng * math.cos(self.roll) * math.cos(self.pitch)
        H = np.array([[1.0, 0.0, 0.0]])
        self._kf_update(self.xv, self.Pv, np.array([-h]), H, np.array([[std * std]]), 5.0)

    def update_baro(self, alt: float, std: float = 0.6) -> None:
        H = np.array([[1.0, 0.0, 0.0]])
        self._kf_update(self.xv, self.Pv, np.array([-alt]), H, np.array([[std * std]]), 6.0)

    def update_heading(self, yaw_meas: float, std: float) -> bool:
        y = wrap_pi(yaw_meas - self.xy[0])
        S = float(self.Py[0, 0]) + std * std
        if y * y / S > 16.0:
            return False
        K = self.Py[:, 0] / S
        self.xy += K * y
        self.xy[0] = wrap_pi(self.xy[0])
        self.Py = self.Py - np.outer(K, self.Py[0, :])
        return True

    def update_external_vision(self, t: float, pos_xy: np.ndarray, std: float,
                               yaw: float | None = None, yaw_std: float = 0.05) -> bool:
        """Landmark / VIO position fix (MAVLink VISION_POSITION_ESTIMATE)."""
        if not self.use_ev:
            return False
        std = max(std, self.cfg["ev_min_std_m"])
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        ok = self._kf_update(self.xh, self.Ph, np.asarray(pos_xy, float), H,
                             np.eye(2) * std * std, self.cfg["ev_gate_sigma"])
        if ok:
            self.ev_accepted += 1
            self.last_ev_t = t
            if yaw is not None:
                self.update_heading(yaw, yaw_std)
        else:
            self.ev_rejected += 1
        return ok
