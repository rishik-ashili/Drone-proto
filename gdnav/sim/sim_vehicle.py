"""SimVehicle: the simulated drone + Pixhawk behind the VehicleInterface.

Physics, sensors, the EKF2 analogue and the PX4 controller analogue all run
here at their own rates. A second "dead-reckoning only" estimator is fed the
exact same sensor data but never receives landmark fixes; it exists purely to
show how far the drone would drift without vision correction.
"""
from __future__ import annotations

import math

import numpy as np

from gdnav.interface.vehicle import CameraFrame, Telemetry, VehicleInterface
from gdnav.sim.camera import DownwardCamera
from gdnav.sim.estimator import PositionEstimator
from gdnav.sim.flight_controller import FlightController, FlightMode, Setpoint
from gdnav.sim.quadrotor import Quadrotor
from gdnav.sim.sensors import SensorSuite
from gdnav.sim.world import World

FAULTS = {
    "camera_covered": "Camera covered (lens blocked)",
    "camera_dropout": "Camera frames dropped",
    "dim_lights": "Lights dimmed to 25 %",
    "link_loss": "Companion <-> Pixhawk link lost",
    "featureless_floor": "Featureless floor (optical flow fails)",
    "wind_gust": "Strong wind gust",
    "low_battery": "Battery drops to 22 %",
    "false_landmark": "Inject false-positive detections",
}


class SimVehicle(VehicleInterface):
    def __init__(self, cfg: dict, world: World, start_pos=(0.0, 0.0), start_yaw: float = 0.0):
        self.cfg = cfg
        self.world = world
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        r = cfg["rates_hz"]
        self.dt = 1.0 / r["physics"]
        self.flow_div = max(1, r["physics"] // r["flow"])
        self.cam_period = 1.0 / r["camera"]
        self.depth_period = 1.0 / r["depth"]

        self.quad = Quadrotor(cfg["vehicle"], pos=(start_pos[0], start_pos[1], 0.0), yaw=start_yaw)
        self.sensors = SensorSuite(cfg["sensors"], world, self.rng)
        self.ekf = PositionEstimator(cfg["estimator"], use_external_vision=True, yaw0=start_yaw)
        self.dr = PositionEstimator(cfg["estimator"], use_external_vision=False, yaw0=start_yaw)
        for est in (self.ekf, self.dr):
            est.xh[0:2] = start_pos
        v = cfg["vehicle"]
        self.fc = FlightController(cfg["flight_controller"], v["mass_kg"], v["max_thrust_n"], v["max_tilt_deg"])
        c = cfg["sensors"]["camera"]
        self.camera = DownwardCamera(world, c["width"], c["height"], c["hfov_deg"], self.rng)

        self.t = 0.0
        self._step = 0
        self._next_cam = 0.0
        self._next_depth = 0.0
        self._next_mag = 0.0
        self._range = 0.0
        self._flow_quality = 0
        self.frame: CameraFrame | None = None
        self.frame_count = 0
        self.depth: tuple[np.ndarray, np.ndarray] | None = None
        self.wind = np.array(cfg["wind"]["mean"], float)
        self._gust = np.zeros(2)
        self.faults: dict[str, float] = {}   # name -> sim time injected
        self.collisions: list[dict] = []
        self._in_contact: set[str] = set()
        self._telem = Telemetry()
        self._update_telemetry()

    # ------------------------------------------------------------ faults
    def inject(self, name: str) -> None:
        if name not in FAULTS:
            raise ValueError(f"unknown fault {name}")
        self.faults[name] = self.t
        if name == "low_battery":
            self.quad.battery.set_remaining(min(self.quad.battery.remaining, 0.22))
        self.sensors.flow_degraded = "featureless_floor" in self.faults

    def clear(self, name: str) -> None:
        self.faults.pop(name, None)
        self.sensors.flow_degraded = "featureless_floor" in self.faults

    @property
    def link_up(self) -> bool:
        return "link_loss" not in self.faults

    # --------------------------------------------------------- simulation
    def advance(self, duration: float) -> None:
        n = max(1, int(round(duration / self.dt)))
        for _ in range(n):
            self._physics_step()
        if self.link_up:
            self._update_telemetry()

    def _physics_step(self) -> None:
        dt = self.dt
        q, s = self.quad, self.sensors
        # wind: mean + low-pass gusts (+ injected gust)
        wc = self.cfg["wind"]
        self._gust += (-self._gust * 0.5 * dt) + self.rng.normal(0, wc["gust_std"] * math.sqrt(dt), 2)
        self.wind = np.array(wc["mean"], float) + self._gust
        if "wind_gust" in self.faults:
            self.wind = self.wind + np.array(wc["gust_injection"])

        # flight controller acts on the estimate, not the truth
        thrust, roll, pitch, yaw_rate = self.fc.update(
            self.t, dt, self.ekf.pos, self.ekf.vel, self.ekf.yaw, q.on_ground, q.battery.remaining)
        # Tilt commands are applied in the real body frame, so any heading error
        # in the estimate rotates the resulting acceleration - exactly as on a real drone.
        q.step(dt, thrust, roll, pitch, yaw_rate, self.wind, self.fc.armed)

        self._check_collisions()
        imu = s.imu(self.t, q, dt)
        att_noise = self.rng.normal(0, math.radians(0.3), 2)
        for est in (self.ekf, self.dr):
            est.predict(imu.accel, imu.gyro, q.att[0] + att_noise[0], q.att[1] + att_noise[1], dt)
            if q.on_ground:
                est.reset_on_ground()

        if self._step % self.flow_div == 0:
            flow = s.flow(self.t, q)
            self._flow_quality = flow.quality
            self._range = s.range(q)
            baro = s.baro(q, dt * self.flow_div)
            h = max(self._range, 0.05)
            fstd = self.cfg["sensors"]["flow"]["noise_base"] + self.cfg["sensors"]["flow"]["noise_per_m"] * h
            for est in (self.ekf, self.dr):
                if q.on_ground:
                    est.update_flow(self.t, np.zeros(2), 0.02)  # zero-velocity constraint at rest
                elif flow.valid:
                    est.update_flow(self.t, flow.vel_body, fstd * 1.5)
                est.update_range(self._range)
                est.update_baro(baro)
        if self.t >= self._next_mag:
            self._next_mag = self.t + 0.1
            m = s.mag_heading(q)
            for est in (self.ekf, self.dr):
                est.update_heading(m, math.radians(6.0))

        if self.t >= self._next_cam:
            self._next_cam += self.cam_period
            self._capture()
        if self.t >= self._next_depth:
            self._next_depth += self.depth_period
            self.depth = s.depth_scan(q)

        self.t += dt
        self._step += 1

    def _check_collisions(self, arm_radius: float = 0.25) -> None:
        q = self.quad
        for ob in self.world.obstacles:
            hit = (-q.p[2] < ob.height and
                   math.hypot(q.p[0] - ob.x, q.p[1] - ob.y) < ob.radius + arm_radius)
            if hit and ob.name not in self._in_contact:
                self.collisions.append({"t": round(self.t, 2), "obstacle": ob.name})
                self.fc.events.append(("error", f"COLLISION with {ob.name}"))
            (self._in_contact.add if hit else self._in_contact.discard)(ob.name)

    def _capture(self) -> None:
        if "camera_dropout" in self.faults:
            return
        q = self.quad
        light = 0.25 if "dim_lights" in self.faults else 1.0
        noise = 9.0 if "dim_lights" in self.faults else 3.0
        cam_pos = q.p.copy()
        cam_pos[2] = min(cam_pos[2], -0.12)  # lens height above the floor when landed
        img = self.camera.render(cam_pos, q.att, q.v, light=light, noise=noise)
        if "camera_covered" in self.faults:
            img = (img * 0.04).astype(np.uint8) + self.rng.integers(0, 6, img.shape, dtype=np.uint8)
        # companion receives the frame with the *estimated* attitude / range
        self.frame = CameraFrame(self.t, img, self.ekf.att.copy(), self._range)
        self.frame_count += 1

    def _update_telemetry(self) -> None:
        q = self.quad
        self._telem = Telemetry(
            t=self.t, received_t=self.t, pos=self.ekf.pos.copy(), vel=self.ekf.vel.copy(),
            att=self.ekf.att.copy(), pos_std=self.ekf.pos_std, range_m=self._range,
            armed=self.fc.armed, mode=self.fc.mode, landed=q.on_ground,
            battery=q.battery.remaining, voltage=q.battery.voltage,
            flow_quality=self._flow_quality, failsafe=self.fc.failsafe_reason)

    # ---------------------------------------------------- VehicleInterface
    def now(self) -> float:
        return self.t

    def telemetry(self) -> Telemetry:
        return self._telem

    def arm(self) -> bool:
        return self.link_up and self.fc.arm()

    def start_offboard(self) -> bool:
        return self.link_up and self.fc.start_offboard(self.t)

    def send_setpoint(self, sp: Setpoint) -> None:
        if self.link_up:
            self.fc.set_setpoint(sp, self.t)

    def land(self) -> None:
        if self.link_up:
            self.fc.land()

    def hold(self) -> None:
        if self.link_up:
            self.fc.hold(self.ekf.pos, self.ekf.yaw)

    def send_vision_position(self, t, pos_xy, std, yaw=None) -> bool:
        if not self.link_up:
            return False
        return self.ekf.update_external_vision(self.t, pos_xy, std, yaw, yaw_std=math.radians(1.5))

    def latest_frame(self) -> CameraFrame | None:
        return self.frame

    def depth_scan(self):
        return self.depth

    # ------------------------------------------- ground truth (sim only)
    def truth(self) -> dict:
        q = self.quad
        return {"pos": q.p.copy(), "vel": q.v.copy(), "att": q.att.copy(),
                "dr_pos": self.dr.pos.copy(), "wind": self.wind.copy(),
                "distance": q.distance_flown, "current": q.battery.current}

    @property
    def flying(self) -> bool:
        return self.fc.armed and self.fc.mode != FlightMode.DISARMED
