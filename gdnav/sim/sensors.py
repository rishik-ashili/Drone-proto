"""Sensor models: IMU, optical flow, rangefinder, barometer, magnetometer and
a forward depth sensor. Noise, biases and scale errors are what make a
GPS-denied position estimate drift - and what the landmark correction fixes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gdnav.common.geometry import wrap_pi
from gdnav.sim.quadrotor import Quadrotor
from gdnav.sim.world import World


@dataclass
class ImuSample:
    t: float
    accel: np.ndarray  # body specific force, m/s^2
    gyro: np.ndarray   # body rates, rad/s


@dataclass
class FlowSample:
    t: float
    vel_body: np.ndarray  # metric body-frame horizontal velocity (flow * range)
    quality: int          # 0..255 like PX4 OPTICAL_FLOW quality
    valid: bool


class SensorSuite:
    def __init__(self, cfg: dict, world: World, rng: np.random.Generator):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        imu = cfg["imu"]
        self.accel_bias = rng.normal(0, imu["accel_bias_init"], 3)
        self.accel_bias[2] *= 0.5
        self.gyro_bias = rng.normal(0, imu["gyro_bias_init"], 3)
        fl = cfg["flow"]
        self.flow_scale = 1.0 + rng.uniform(-fl["scale_error"], fl["scale_error"], 2)
        self.baro_offset = 0.0
        self.mag_bias = math.radians(cfg["mag"]["bias_deg"]) * (1 if rng.random() > 0.5 else -1)
        # fault flags set by the failure injector
        self.flow_degraded = False

    def imu(self, t: float, quad: Quadrotor, dt: float) -> ImuSample:
        c = self.cfg["imu"]
        self.accel_bias += self.rng.normal(0, c["accel_bias_rw"] * math.sqrt(dt), 3)
        acc = quad.specific_force_body() + self.accel_bias + self.rng.normal(0, c["accel_noise"], 3)
        gyro = quad.rates + self.gyro_bias + self.rng.normal(0, c["gyro_noise"], 3)
        return ImuSample(t, acc, gyro)

    def flow(self, t: float, quad: Quadrotor) -> FlowSample:
        c = self.cfg["flow"]
        h = -quad.p[2]
        yaw = quad.att[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        vb = np.array([cy * quad.v[0] + sy * quad.v[1], -sy * quad.v[0] + cy * quad.v[1]])
        quality = 200 if h > 0.15 else 0
        if h > c["max_range_m"]:
            quality = 40
        if self.flow_degraded:  # featureless floor / poor light
            quality = int(self.rng.integers(0, 45))
        valid = quality >= c["min_quality"]
        noise = c["noise_base"] + c["noise_per_m"] * max(h, 0.0)
        meas = vb * self.flow_scale + self.rng.normal(0, noise, 2)
        if self.flow_degraded:
            meas += self.rng.normal(0, 0.6, 2)
        return FlowSample(t, meas, quality, valid)

    def range(self, quad: Quadrotor) -> float:
        c = self.cfg["rangefinder"]
        r, p = quad.att[0], quad.att[1]
        d = max(0.0, -quad.p[2]) / max(0.3, math.cos(r) * math.cos(p))
        return float(np.clip(d + self.rng.normal(0, c["noise"]), 0.0, c["max_range_m"]))

    def baro(self, quad: Quadrotor, dt: float) -> float:
        c = self.cfg["baro"]
        self.baro_offset += self.rng.normal(0, c["drift_per_s"] * math.sqrt(dt))
        return float(-quad.p[2] + self.baro_offset + self.rng.normal(0, c["noise"]))

    def mag_heading(self, quad: Quadrotor) -> float:
        c = self.cfg["mag"]
        return wrap_pi(quad.att[2] + self.mag_bias + self.rng.normal(0, math.radians(c["noise_deg"])))

    def depth_scan(self, quad: Quadrotor) -> tuple[np.ndarray, np.ndarray]:
        """Forward depth sensor (stereo depth stand-in): returns (angles_body, ranges)."""
        c = self.cfg["depth"]
        n = c["rays"]
        half = math.radians(c["fov_deg"]) / 2
        angles = np.linspace(-half, half, n)
        ranges = np.full(n, c["max_range_m"], float)
        px, py, alt = quad.p[0], quad.p[1], -quad.p[2]
        for i, a in enumerate(angles):
            th = quad.att[2] + a
            dx, dy = math.cos(th), math.sin(th)
            for ob in self.world.obstacles:
                if alt > ob.height:
                    continue
                fx, fy = px - ob.x, py - ob.y
                b = fx * dx + fy * dy
                cc = fx * fx + fy * fy - ob.radius ** 2
                disc = b * b - cc
                if disc < 0:
                    continue
                s = -b - math.sqrt(disc)
                if 0 < s < ranges[i]:
                    ranges[i] = s
        ranges = np.where(ranges < c["max_range_m"], ranges + self.rng.normal(0, c["noise"], n), ranges)
        return angles, ranges
