"""PX4 multicopter controller analogue (runs "on the Pixhawk").

Implements just enough of PX4's behaviour for the companion software to be
developed against it honestly:
  * arming, Offboard mode that only engages while a setpoint stream exists
  * cascaded position -> velocity -> acceleration -> attitude control that
    acts on the *estimated* state (so estimator drift becomes real drift)
  * offboard-loss failsafe after COM_OF_LOSS_T (hold or land)
  * auto-land and a critical-battery land failsafe
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gdnav.common.geometry import wrap_pi

G = 9.81


@dataclass
class Setpoint:
    """Offboard setpoint, either position or velocity in local NED."""
    kind: str = "position"            # position | velocity
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float | None = None
    yaw_rate: float = 0.0


class FlightMode:
    DISARMED = "DISARMED"
    ARMED = "ARMED"            # armed, on ground, waiting for a mode
    OFFBOARD = "OFFBOARD"
    HOLD = "HOLD"              # position hold on the local estimate
    LAND = "LAND"


class FlightController:
    def __init__(self, cfg: dict, mass: float, max_thrust: float, max_tilt_deg: float):
        self.p = cfg
        self.m = mass
        self.max_thrust = max_thrust
        self.max_tilt = math.radians(max_tilt_deg)
        self.mode = FlightMode.DISARMED
        self.armed = False
        self.sp: Setpoint | None = None
        self.last_sp_t = -1e9
        self.hold_pos: np.ndarray | None = None
        self.hold_yaw = 0.0
        self.vel_int = np.zeros(3)
        self.prev_vel_err = np.zeros(3)
        self.events: list[tuple[str, str]] = []
        self.failsafe_reason = ""

    # -------------------------------------------------------------- commands
    def arm(self) -> bool:
        if self.mode == FlightMode.DISARMED:
            self.armed = True
            self.mode = FlightMode.ARMED
            self.events.append(("info", "Armed"))
            return True
        return self.armed

    def disarm(self) -> None:
        self.armed = False
        self.mode = FlightMode.DISARMED
        self.vel_int[:] = 0
        self.events.append(("info", "Disarmed"))

    def set_setpoint(self, sp: Setpoint, t: float) -> None:
        self.sp = sp
        self.last_sp_t = t

    def start_offboard(self, t: float) -> bool:
        # PX4 refuses Offboard unless setpoints are already streaming
        if not self.armed or t - self.last_sp_t > 0.5:
            self.events.append(("warn", "Offboard rejected: no setpoint stream"))
            return False
        self.mode = FlightMode.OFFBOARD
        self.failsafe_reason = ""
        self.events.append(("info", "Offboard mode engaged"))
        return True

    def land(self) -> None:
        if self.armed:
            self.mode = FlightMode.LAND
            self.events.append(("info", "Land mode"))

    def hold(self, est_pos: np.ndarray, est_yaw: float) -> None:
        self.mode = FlightMode.HOLD
        self.hold_pos = est_pos.copy()
        self.hold_yaw = est_yaw

    # ------------------------------------------------------------ control
    def update(self, t: float, dt: float, pos: np.ndarray, vel: np.ndarray, yaw: float,
               on_ground: bool, battery: float):
        """Returns (thrust_N, roll_cmd, pitch_cmd, yaw_rate_cmd)."""
        if not self.armed:
            return 0.0, 0.0, 0.0, 0.0

        # ---- failsafes (PX4 commander)
        if self.mode == FlightMode.OFFBOARD and t - self.last_sp_t > self.p["COM_OF_LOSS_T"]:
            self.failsafe_reason = "offboard setpoints lost"
            if self.p["COM_OBL_RC_ACT"] == "land":
                self.mode = FlightMode.LAND
            else:
                self.hold(pos, yaw)
            self.events.append(("error", f"Failsafe: offboard lost -> {self.mode}"))
        if battery < self.p["BAT_CRIT_THR"] and self.mode != FlightMode.LAND and not on_ground:
            self.failsafe_reason = "battery critical"
            self.mode = FlightMode.LAND
            self.events.append(("error", "Failsafe: battery critical -> LAND"))

        # ---- choose velocity target
        yaw_rate = 0.0
        if self.mode == FlightMode.ARMED:
            return 0.0, 0.0, 0.0, 0.0
        if self.mode == FlightMode.OFFBOARD and self.sp is not None:
            sp = self.sp
            if sp.kind == "position":
                vel_sp = self._pos_to_vel(sp.pos, pos)
            else:
                vel_sp = sp.vel.copy()
            if sp.yaw is not None:
                yaw_rate = self.p["MPC_YAWRAUTO_MAX"] * math.tanh(2.0 * wrap_pi(sp.yaw - yaw))
            else:
                yaw_rate = sp.yaw_rate
        elif self.mode == FlightMode.HOLD:
            vel_sp = self._pos_to_vel(self.hold_pos, pos)
            yaw_rate = 1.0 * wrap_pi(self.hold_yaw - yaw)
        elif self.mode == FlightMode.LAND:
            vel_sp = np.array([-0.5 * vel[0], -0.5 * vel[1], self.p["MPC_LAND_SPEED"]])
            if on_ground:
                self.disarm()
                self.events.append(("info", "Landed"))
                return 0.0, 0.0, 0.0, 0.0
        else:
            vel_sp = np.zeros(3)

        vmax = self.p["MPC_XY_VEL_MAX"]
        n = np.linalg.norm(vel_sp[:2])
        if n > vmax:
            vel_sp[:2] *= vmax / n
        vel_sp[2] = float(np.clip(vel_sp[2], -self.p["MPC_Z_VEL_MAX_UP"], self.p["MPC_Z_VEL_MAX_DN"]))

        # ---- velocity PID -> acceleration
        err = vel_sp - vel
        kp = np.array([self.p["MPC_XY_VEL_P"]] * 2 + [self.p["MPC_Z_VEL_P"]])
        ki = np.array([self.p["MPC_XY_VEL_I"]] * 2 + [self.p["MPC_Z_VEL_I"]])
        kd = np.array([self.p["MPC_XY_VEL_D"]] * 2 + [0.0])
        if not on_ground:
            self.vel_int = np.clip(self.vel_int + err * dt, -2.0, 2.0)
        d_err = (err - self.prev_vel_err) / dt if dt > 0 else 0.0
        self.prev_vel_err = err
        acc = kp * err + ki * self.vel_int + kd * d_err

        # ---- acceleration -> thrust and attitude
        az = float(np.clip(acc[2], -6.0, 6.0))
        thrust = self.m * (G - az)
        cy, sy = math.cos(yaw), math.sin(yaw)
        a_fwd = cy * acc[0] + sy * acc[1]
        a_right = -sy * acc[0] + cy * acc[1]
        denom = max(2.0, G - az)
        pitch = -math.atan2(a_fwd, denom)
        roll = math.atan2(a_right, denom)
        pitch = float(np.clip(pitch, -self.max_tilt, self.max_tilt))
        roll = float(np.clip(roll, -self.max_tilt, self.max_tilt))
        thrust /= max(0.5, math.cos(roll) * math.cos(pitch))
        return float(np.clip(thrust, 0.0, self.max_thrust)), roll, pitch, yaw_rate

    def _pos_to_vel(self, target: np.ndarray, pos: np.ndarray) -> np.ndarray:
        e = target - pos
        v = np.array([self.p["MPC_XY_P"] * e[0], self.p["MPC_XY_P"] * e[1], self.p["MPC_Z_P"] * e[2]])
        return v
