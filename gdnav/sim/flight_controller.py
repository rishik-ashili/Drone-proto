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
    """Flight controller mode names (subset of PX4 navigation states)."""
    DISARMED = "DISARMED"
    ARMED = "ARMED"            # armed, on ground, waiting for a mode
    OFFBOARD = "OFFBOARD"
    HOLD = "HOLD"              # position hold on the local estimate
    LAND = "LAND"
    POSCTL = "POSCTL"          # pilot Position mode: sticks command velocity, centred sticks hold position


class FlightController:
    """PX4 multicopter controller analogue: modes, cascaded control, failsafes, collision prevention."""
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
        self.sticks = np.zeros(4)      # [forward, right, up, yaw] in -1..1 (RC / gamepad)
        self.sticks_t = -1e9
        self._pos_lock: np.ndarray | None = None
        self._z_lock: float | None = None
        self._ground_since: float | None = None
        self.obstacle_scan = None     # (angles_body, ranges) from the distance sensor
        self._yaw_lock = 0.0

    # -------------------------------------------------------------- commands
    def arm(self) -> bool:
        """Arm if disarmed."""
        if self.mode == FlightMode.DISARMED:
            self.armed = True
            self.mode = FlightMode.ARMED
            self.events.append(("info", "Armed"))
            return True
        return self.armed

    def disarm(self) -> None:
        """Disarm and reset integrators."""
        self.armed = False
        self.mode = FlightMode.DISARMED
        self.vel_int[:] = 0
        self.events.append(("info", "Disarmed"))

    def set_setpoint(self, sp: Setpoint, t: float) -> None:
        """Store the latest Offboard setpoint and its arrival time."""
        self.sp = sp
        self.last_sp_t = t

    def start_offboard(self, t: float) -> bool:
        # PX4 refuses Offboard unless setpoints are already streaming
        """Engage Offboard only while setpoints are streaming (as PX4 requires)."""
        if not self.armed or t - self.last_sp_t > 0.5:
            self.events.append(("warn", "Offboard rejected: no setpoint stream"))
            return False
        self.mode = FlightMode.OFFBOARD
        self.failsafe_reason = ""
        self.events.append(("info", "Offboard mode engaged"))
        return True

    def land(self) -> None:
        """Switch to Land mode."""
        if self.armed:
            self.mode = FlightMode.LAND
            self.events.append(("info", "Land mode"))

    def position_mode(self, est_pos: np.ndarray, est_yaw: float) -> None:
        """Pilot takes over (RC mode switch). Works without GPS: Position mode only needs
        the local position estimate (optical flow / VIO)."""
        if not self.armed:
            self.arm()
        self.mode = FlightMode.POSCTL
        self._pos_lock = est_pos.copy()
        self._z_lock = float(est_pos[2]) if est_pos[2] < -0.3 else None
        self._yaw_lock = est_yaw
        self.events.append(("info", "Position mode (pilot)"))

    def set_sticks(self, sticks, t: float) -> None:
        """Pilot stick input [forward, right, up, yaw] in -1..1."""
        self.sticks = np.clip(np.asarray(sticks, float), -1.0, 1.0)
        self.sticks_t = t

    def hold(self, est_pos: np.ndarray, est_yaw: float) -> None:
        """Switch to Hold at the current estimated position and heading."""
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
        elif self.mode == FlightMode.POSCTL:
            st = self.sticks if t - self.sticks_t < 0.5 else np.zeros(4)  # stale input = centred sticks
            st = np.where(np.abs(st) < 0.05, 0.0, st)
            cy, sy = math.cos(yaw), math.sin(yaw)
            vmax = self.p["MPC_XY_VEL_MAX"] * 0.8
            v_fwd, v_right = st[0] * vmax, st[1] * vmax
            vel_sp = np.array([cy * v_fwd - sy * v_right, sy * v_fwd + cy * v_right,
                               -st[2] * self.p["MPC_Z_VEL_MAX_UP"]])
            if on_ground and st[2] <= 0:
                if self._ground_since is None:
                    self._ground_since = t
                elif t - self._ground_since > 2.0:      # COM_DISARM_LAND
                    self.disarm()
                    self.events.append(("info", "Landed (auto-disarm)"))
                return 0.0, 0.0, 0.0, 0.0
            self._ground_since = None
            if abs(st[0]) + abs(st[1]) == 0.0:   # sticks centred: brake, then hold the position
                if self._pos_lock is None and float(np.linalg.norm(vel[:2])) < 0.3:
                    self._pos_lock = pos.copy()
                vel_sp[:2] = self._pos_to_vel(self._pos_lock, pos)[:2] if self._pos_lock is not None else 0.0
            else:
                self._pos_lock = None
            if st[2] == 0.0:                      # throttle centred: hold altitude
                if self._z_lock is None:
                    self._z_lock = float(pos[2])
                vel_sp[2] = self.p["MPC_Z_P"] * (self._z_lock - pos[2])
            else:
                self._z_lock = None
            yaw_rate = st[3] * self.p["MPC_YAWRAUTO_MAX"]
            vel_sp[:2] = self._collision_prevention(vel_sp[:2], yaw)
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

    def _collision_prevention(self, v_xy: np.ndarray, yaw: float) -> np.ndarray:
        """PX4 Collision Prevention (CP_DIST): limit the pilot's velocity towards obstacles seen
        by the distance sensor so the drone stops CP_DIST short of them."""
        cp = self.p.get("CP_DIST", -1.0)
        if cp is None or cp <= 0 or self.obstacle_scan is None:
            return v_xy
        angles, ranges = self.obstacle_scan
        v = np.array(v_xy, float)
        for a, r in zip(angles, ranges):
            if r >= 5.9:
                continue
            u = np.array([math.cos(yaw + a), math.sin(yaw + a)])
            allowed = max(0.0, r - cp) * 0.8      # m/s towards the obstacle, braking to zero at CP_DIST
            comp = float(v @ u)
            if comp > allowed:
                v -= (comp - allowed) * u
        return v

    def _pos_to_vel(self, target: np.ndarray, pos: np.ndarray) -> np.ndarray:
        e = target - pos
        v = np.array([self.p["MPC_XY_P"] * e[0], self.p["MPC_XY_P"] * e[1], self.p["MPC_Z_P"] * e[2]])
        return v
