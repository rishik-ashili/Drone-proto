"""Rigid-body quadrotor model: thrust + attitude response + drag + wind + battery.

The inner attitude/rate loops of PX4 are fast compared with the position
loop, so they are modelled as a first-order response of roll / pitch to the
commanded angles. Thrust acts along the body -z axis.
"""
from __future__ import annotations

import math

import numpy as np

from gdnav.common.geometry import rot_from_euler, wrap_pi

G = 9.81


class Battery:
    def __init__(self, capacity_mah: float, cells: int):
        self.capacity_ah = capacity_mah / 1000.0
        self.cells = cells
        self.used_ah = 0.0
        self.current = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, 1.0 - self.used_ah / self.capacity_ah)

    @property
    def voltage(self) -> float:
        # simple LiPo curve 4.2 V -> 3.5 V per cell plus internal resistance sag
        soc = self.remaining
        v_cell = 3.5 + 0.55 * soc + 0.15 * soc ** 4
        return self.cells * v_cell - 0.012 * self.current

    def drain(self, current_a: float, dt: float) -> None:
        self.current = current_a
        self.used_ah += current_a * dt / 3600.0

    def set_remaining(self, frac: float) -> None:
        self.used_ah = self.capacity_ah * (1.0 - frac)


class Quadrotor:
    def __init__(self, cfg: dict, pos=(0.0, 0.0, 0.0), yaw: float = 0.0):
        self.m = cfg["mass_kg"]
        self.max_thrust = cfg["max_thrust_n"]
        self.drag = cfg["drag_coeff"]
        self.tau = cfg["attitude_tau_s"]
        self.max_tilt = math.radians(cfg["max_tilt_deg"])
        self.hover_current = cfg["hover_current_a"]
        self.battery = Battery(cfg["battery_capacity_mah"], cfg["battery_cells"])

        self.p = np.array(pos, float)
        self.v = np.zeros(3)
        self.att = np.array([0.0, 0.0, yaw])  # roll, pitch, yaw
        self.rates = np.zeros(3)              # body rates (approx.)
        self.acc = np.zeros(3)                # world acceleration
        self.thrust = 0.0
        self.on_ground = True
        self.distance_flown = 0.0

    @property
    def hover_thrust(self) -> float:
        return self.m * G

    def specific_force_body(self) -> np.ndarray:
        """What an accelerometer measures: R_bw (a - g)."""
        R = rot_from_euler(*self.att)
        return R.T @ (self.acc - np.array([0.0, 0.0, G]))

    def step(self, dt: float, thrust: float, roll_cmd: float, pitch_cmd: float,
             yaw_rate_cmd: float, wind: np.ndarray, armed: bool) -> None:
        if not armed:
            thrust = 0.0
        thrust = float(np.clip(thrust, 0.0, self.max_thrust))
        roll_cmd = float(np.clip(roll_cmd, -self.max_tilt, self.max_tilt))
        pitch_cmd = float(np.clip(pitch_cmd, -self.max_tilt, self.max_tilt))

        prev = self.att.copy()
        a = dt / (self.tau + dt)
        if self.on_ground and thrust < self.hover_thrust:
            self.att[0] *= 0.9
            self.att[1] *= 0.9
        else:
            self.att[0] += a * (roll_cmd - self.att[0])
            self.att[1] += a * (pitch_cmd - self.att[1])
            self.att[2] = wrap_pi(self.att[2] + yaw_rate_cmd * dt)
        self.rates = np.array([(self.att[0] - prev[0]) / dt, (self.att[1] - prev[1]) / dt,
                               wrap_pi(self.att[2] - prev[2]) / dt])

        R = rot_from_euler(*self.att)
        f_thrust = R @ np.array([0.0, 0.0, -thrust])
        rel = self.v.copy()
        rel[:2] -= wind
        acc = f_thrust / self.m + np.array([0.0, 0.0, G]) - (self.drag / self.m) * rel

        if self.on_ground:
            if acc[2] < 0:  # enough thrust to lift off
                self.on_ground = False
            else:
                acc[:] = 0.0
                self.v[:] = 0.0

        self.acc = acc
        old = self.p.copy()
        self.v += acc * dt
        self.p += self.v * dt
        if self.p[2] >= 0.0 and not self.on_ground and self.v[2] >= 0:
            self.p[2] = 0.0
            self.v[:] = 0.0
            self.acc[:] = 0.0
            self.on_ground = True
        self.distance_flown += float(np.linalg.norm(self.p[:2] - old[:2]))

        # electrical model: current grows with thrust^1.5 relative to hover
        rel_thrust = thrust / self.hover_thrust
        current = 0.6 + self.hover_current * rel_thrust ** 1.5 if armed else 0.3
        self.battery.drain(current, dt)
