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
    """Vehicle state as the companion sees it (estimates, never ground truth)."""
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
    """One downward-camera image with the attitude, range and position estimate at exposure."""
    t: float
    image: np.ndarray
    # vehicle state at exposure time, as known by the companion (estimate)
    att: np.ndarray
    range_m: float
    pos: np.ndarray | None = None   # estimated position at exposure (latency compensation)


class VehicleInterface(ABC):
    """The one API mission, perception and safety code use; implemented by the simulator, MAVSDK and ROS 2 backends."""
    def now(self) -> float:
        """Vehicle clock (s) shared by telemetry timestamps, frames and the nav loop."""
        return time.monotonic()

    @abstractmethod
    def telemetry(self) -> Telemetry:
        """Latest vehicle state: estimated position/velocity/attitude, mode, armed, battery, range."""

    @abstractmethod
    def arm(self) -> bool:
        """Arm the motors. Returns False if the flight controller refused."""

    @abstractmethod
    def start_offboard(self) -> bool:
        """Switch to Offboard mode (setpoints must already be streaming)."""

    @abstractmethod
    def send_setpoint(self, sp: Setpoint) -> None:
        """Stream one Offboard position or velocity setpoint (local NED)."""

    @abstractmethod
    def land(self) -> None:
        """Command the flight controller's Land mode."""

    @abstractmethod
    def hold(self) -> None:
        """Command the flight controller's Hold (position hold) mode."""

    @abstractmethod
    def send_vision_position(self, t: float, pos_xy: np.ndarray, std: float,
                             yaw: float | None = None) -> bool:
        """Forward a landmark (or VIO) position fix to the flight controller's
        estimator. Returns True if the estimator accepted it."""

    def send_odometry(self, t: float, pos_xy: np.ndarray, yaw: float, vel_xy: np.ndarray | None,
                      pos_std: float, yaw_std: float, vel_std: float, reset_counter: int = 0) -> bool:
        """Full external-vision odometry (VIO): position + heading + velocity.
        Backends without a velocity channel fall back to a position fix."""
        return self.send_vision_position(t, pos_xy, pos_std, yaw)

    def position_mode(self) -> bool:
        """Switch to pilot Position mode (as flipping the RC mode switch would)."""
        return False

    def manual_control(self, sticks) -> None:
        """Pilot stick input [forward, right, up, yaw], each -1..1."""

    def vio_samples(self) -> list:
        """VIO odometry samples since the last call (simulator; on the Jetson they come from ROS 2)."""
        return []

    def latest_frame(self) -> CameraFrame | None:
        """Most recent camera frame, or None when the backend has no camera."""
        return None

    def depth_scan(self):
        """(angles_body_rad, ranges_m) from a forward depth sensor, or None."""
        return None

    def close(self) -> None:
        """Release connections and threads."""
        pass
