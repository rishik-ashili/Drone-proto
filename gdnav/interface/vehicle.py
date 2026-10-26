"""px4_interface: the one API the companion software uses to talk to a vehicle.

Two implementations exist:
  * ``gdnav.sim.sim_vehicle.SimVehicle``  - built-in physics / EKF / PX4 analogue
  * ``gdnav.interface.px4_mavsdk.Px4MavsdkVehicle`` - real PX4 (SITL or Pixhawk)
    over MAVSDK-Python Offboard

Mission logic, perception and safety code never know which one they talk to.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from gdnav.sim.flight_controller import Setpoint


@dataclass
class Telemetry:
    t: float = 0.0                 # vehicle time of the sample (s)
    received_t: float = 0.0        # companion time it arrived (link monitoring)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # estimated local NED
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))   # roll, pitch, yaw
    pos_std: float = 0.0           # horizontal 1-sigma (eph)
    range_m: float = 0.0           # downward rangefinder
    armed: bool = False
    mode: str = "DISARMED"
    landed: bool = True
    battery: float = 1.0           # remaining fraction
    voltage: float = 0.0
    flow_quality: int = 0
    failsafe: str = ""


@dataclass
class CameraFrame:
    t: float
    image: np.ndarray
    # vehicle state at exposure time, as known by the companion (estimate)
    att: np.ndarray
    range_m: float


class VehicleInterface(ABC):
    def now(self) -> float:
        """Vehicle clock (s) shared by telemetry timestamps, frames and the nav loop."""
        return time.monotonic()

    @abstractmethod
    def telemetry(self) -> Telemetry: ...

    @abstractmethod
    def arm(self) -> bool: ...

    @abstractmethod
    def start_offboard(self) -> bool: ...

    @abstractmethod
    def send_setpoint(self, sp: Setpoint) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def hold(self) -> None: ...

    @abstractmethod
    def send_vision_position(self, t: float, pos_xy: np.ndarray, std: float,
                             yaw: float | None = None) -> bool:
        """Forward a landmark (or VIO) position fix to the flight controller's
        estimator. Returns True if the estimator accepted it."""

    def latest_frame(self) -> CameraFrame | None:
        return None

    def depth_scan(self):
        """(angles_body_rad, ranges_m) from a forward depth sensor, or None."""
        return None

    def close(self) -> None:
        pass
