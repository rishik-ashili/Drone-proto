"""px4_interface over MAVSDK-Python: PX4 SITL (Gazebo) or a real Pixhawk.

The mission code is synchronous (one tick at the navigation rate); MAVSDK is
asyncio. This adapter runs MAVSDK in a background event loop thread, caches
telemetry, and forwards setpoints / commands - so the exact same
MissionRunner drives the built-in simulator and real PX4.

Offboard control:  set_position_ned / set_velocity_ned (local NED, no GPS)
Landmark fixes:    MAVLink VISION_POSITION_ESTIMATE via the Mocap plugin, fused
                   by EKF2 when EKF2_EV_CTRL enables horizontal position
                   (see config/px4/gps_denied_flow.params)

Connection URLs:   udp://:14540            PX4 SITL (default offboard port)
                   serial:///dev/ttyACM0:921600   Pixhawk over USB / TELEM2

Note: the camera is not provided by PX4. Pass a frame source (USB camera,
Gazebo camera bridge) to ``Px4MavsdkVehicle(frame_source=...)``; without one,
the mission runs on PX4's estimate alone (e.g. the Phase 0 hello_square).
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Callable

import numpy as np

from gdnav.interface.vehicle import CameraFrame, Telemetry, VehicleInterface
from gdnav.sim.flight_controller import Setpoint

try:  # gRPC MAVSDK-Python API: 'mavsdk-grpc' (new name) or 'mavsdk<4' (old name)
    try:
        from mavsdk_grpc import System
        from mavsdk_grpc.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate
        from mavsdk_grpc.offboard import OffboardError, PositionNedYaw, VelocityNedYaw
    except ImportError:
        from mavsdk import System
        from mavsdk.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate
        from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw
except ImportError:  # pragma: no cover - optional dependency
    System = None


class Px4MavsdkVehicle(VehicleInterface):
    def __init__(self, url: str = "udp://:14540", frame_source: Callable[[], np.ndarray | None] | None = None,
                 connect_timeout: float = 30.0):
        if System is None:
            raise ImportError("mavsdk is not installed: pip install -r requirements-px4.txt")
        self.url = url
        self.frame_source = frame_source
        self._telem = Telemetry()
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mavsdk")
        self._thread.start()
        self.drone = System()
        self._run(self._connect(), timeout=connect_timeout)
        for coro in (self._pos_task(), self._att_task(), self._status_task(), self._battery_task(),
                     self._range_task(), self._armed_task(), self._mode_task(), self._landed_task()):
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        if frame_source:
            threading.Thread(target=self._camera_loop, daemon=True, name="camera").start()

    # ------------------------------------------------------------ plumbing
    def _run(self, coro, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _fire(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect(self):
        await self.drone.connect(system_address=self.url)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        print(f"[px4] connected via {self.url}")

    def _upd(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self._telem, k, v)
            self._telem.received_t = time.monotonic() - _T0
            self._telem.t = self._telem.received_t

    async def _pos_task(self):
        async for pv in self.drone.telemetry.position_velocity_ned():
            p, v = pv.position, pv.velocity
            self._upd(pos=np.array([p.north_m, p.east_m, p.down_m]),
                      vel=np.array([v.north_m_s, v.east_m_s, v.down_m_s]))

    async def _att_task(self):
        async for a in self.drone.telemetry.attitude_euler():
            self._upd(att=np.radians([a.roll_deg, a.pitch_deg, a.yaw_deg]))

    async def _status_task(self):
        # estimator position accuracy is not exposed directly; use health flags as a proxy
        async for h in self.drone.telemetry.health():
            self._upd(pos_std=0.1 if h.is_local_position_ok else 5.0)

    async def _battery_task(self):
        async for b in self.drone.telemetry.battery():
            self._upd(battery=float(b.remaining_percent) / (100.0 if b.remaining_percent > 1.0 else 1.0),
                      voltage=float(b.voltage_v))

    async def _range_task(self):
        async for d in self.drone.telemetry.distance_sensor():
            self._upd(range_m=float(d.current_distance_m))

    async def _armed_task(self):
        async for armed in self.drone.telemetry.armed():
            self._upd(armed=bool(armed))

    async def _mode_task(self):
        names = {"OFFBOARD": "OFFBOARD", "HOLD": "HOLD", "LAND": "LAND", "RETURN_TO_LAUNCH": "RTL"}
        async for m in self.drone.telemetry.flight_mode():
            self._upd(mode=names.get(m.name, m.name))

    async def _landed_task(self):
        async for s in self.drone.telemetry.landed_state():
            self._upd(landed=s.name == "ON_GROUND")

    def _camera_loop(self):
        while True:
            img = self.frame_source()
            if img is not None:
                tl = self.telemetry()
                self._frame = CameraFrame(time.monotonic() - _T0, img, tl.att.copy(),
                                          tl.range_m if tl.range_m > 0 else -tl.pos[2])
            time.sleep(1 / 30)

    # ---------------------------------------------------- VehicleInterface
    def now(self) -> float:
        return time.monotonic() - _T0

    def telemetry(self) -> Telemetry:
        with self._lock:
            t = Telemetry(**self._telem.__dict__)
        if t.range_m <= 0:
            t.range_m = max(0.0, -t.pos[2])
        return t

    def arm(self) -> bool:
        try:
            self._run(self.drone.action.arm())
            return True
        except Exception as e:  # noqa: BLE001 - surface PX4's reason
            print(f"[px4] arm failed: {e}")
            return False

    def start_offboard(self) -> bool:
        try:
            self._run(self.drone.offboard.start())
            return True
        except OffboardError as e:
            print(f"[px4] offboard start failed: {e._result.result}")
            return False

    def send_setpoint(self, sp: Setpoint) -> None:
        yaw_deg = math.degrees(sp.yaw) if sp.yaw is not None else math.degrees(self.telemetry().att[2])
        if sp.kind == "position":
            self._fire(self.drone.offboard.set_position_ned(
                PositionNedYaw(float(sp.pos[0]), float(sp.pos[1]), float(sp.pos[2]), yaw_deg)))
        else:
            self._fire(self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2]), yaw_deg)))

    def land(self) -> None:
        self._fire(self.drone.action.land())

    def hold(self) -> None:
        self._fire(self.drone.action.hold())

    def send_vision_position(self, t, pos_xy, std, yaw=None) -> bool:
        tl = self.telemetry()
        yaw_v = float(yaw) if yaw is not None else float(tl.att[2])
        cov = [float("nan")] * 21
        cov[0] = cov[6] = float(std) ** 2         # x, y variance (upper-triangular 6x6, row-major)
        cov[11] = 0.05 ** 2                        # z
        cov[15] = cov[18] = 0.05 ** 2              # roll, pitch
        cov[20] = (0.03 if yaw is not None else 1.0) ** 2
        msg = VisionPositionEstimate(int(time.time() * 1e6),
                                     PositionBody(float(pos_xy[0]), float(pos_xy[1]), float(tl.pos[2])),
                                     AngleBody(float(tl.att[0]), float(tl.att[1]), yaw_v), Covariance(cov), 0)
        self._fire(self.drone.mocap.set_vision_position_estimate(msg))
        return True  # EKF2 gating happens on the Pixhawk; result is not reported back

    def latest_frame(self) -> CameraFrame | None:
        return self._frame

    def close(self) -> None:
        try:
            self._run(self.drone.offboard.stop(), timeout=3)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


_T0 = time.monotonic()
