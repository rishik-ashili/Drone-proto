#!/usr/bin/env python3
"""Bench / pre-flight check of a PX4 vehicle over MAVSDK (Pixhawk on USB, or SITL).

    python scripts/bench_check.py --url serial:///dev/ttyACM0:921600        # read-only checks
    python scripts/bench_check.py --url udp://:14540 --vision-test          # + external-vision fusion
    python scripts/bench_check.py --url ... --offboard-test                 # + arm/offboard (PROPS OFF!)

Checks (P1 exit criterion "setpoints verified on the real controller, props off"):
  1. connection and firmware version
  2. every parameter in config/px4/gps_denied_flow.params matches the vehicle
  3. health: sensor calibration, local position valid WITHOUT GPS
  4. rangefinder, battery and attitude streams present and plausible
  5. (--vision-test) landmark-style VISION_POSITION_ESTIMATE accepted: local position stays valid
  6. (--offboard-test) arm, stream setpoints, engage Offboard for 3 s, disarm
Writes logs/bench_check_<time>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.apply_px4_params import read_params  # noqa: E402

try:
    from mavsdk_grpc import System
    from mavsdk_grpc.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate
    from mavsdk_grpc.offboard import PositionNedYaw
except ImportError:
    from mavsdk import System
    from mavsdk.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate
    from mavsdk.offboard import PositionNedYaw

results: list[dict] = []


def report(name: str, ok: bool | None, detail: str = "") -> None:
    tag = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
    results.append({"check": name, "result": tag, "detail": detail})
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))


async def first(gen, timeout=5.0):
    async def _f():
        async for v in gen:
            return v
    return await asyncio.wait_for(_f(), timeout)


async def run(a) -> None:
    drone = System()
    await drone.connect(system_address=a.url)
    await asyncio.wait_for(_wait_connected(drone), 30)
    try:
        v = await drone.info.get_version()
        report("connection", True, f"PX4 {v.flight_sw_major}.{v.flight_sw_minor}.{v.flight_sw_patch}")
    except Exception as e:  # noqa: BLE001
        report("connection", True, f"version unavailable ({e})")

    # ---- parameters
    bad = []
    for name, want in read_params(Path(a.params)):
        try:
            if "." in want:
                got = await drone.param.get_param_float(name)
                ok = math.isclose(got, float(want), rel_tol=1e-3, abs_tol=1e-4)
            else:
                got = await drone.param.get_param_int(name)
                ok = got == int(want)
            if not ok:
                bad.append(f"{name}={got} (want {want})")
        except Exception:  # noqa: BLE001
            bad.append(f"{name} missing")
    report("parameters match gps_denied_flow.params", not bad, "; ".join(bad[:8]) + (" ..." if len(bad) > 8 else ""))

    # ---- health and streams
    h = await first(drone.telemetry.health())
    report("gyro / accel / mag calibration", h.is_gyrometer_calibration_ok and h.is_accelerometer_calibration_ok
           and h.is_magnetometer_calibration_ok)
    report("local position valid (no GPS)", h.is_local_position_ok,
           "" if h.is_local_position_ok else "EKF2 has no horizontal aiding: flow sensor connected? EKF2_OF_CTRL?")
    report("global position NOT used", not h.is_global_position_ok or a.allow_gps,
           "GPS fix present: EKF2_GPS_CTRL should be 0 for GPS-denied tests" if h.is_global_position_ok else "")
    try:
        d = await first(drone.telemetry.distance_sensor())
        report("rangefinder streaming", d.current_distance_m > 0, f"{d.current_distance_m:.2f} m")
    except asyncio.TimeoutError:
        report("rangefinder streaming", False, "no DISTANCE_SENSOR messages")
    b = await first(drone.telemetry.battery())
    report("battery", b.voltage_v > 0, f"{b.voltage_v:.2f} V, {b.remaining_percent:.0f} %")
    att = await first(drone.telemetry.attitude_euler())
    report("attitude", abs(att.roll_deg) < 10 and abs(att.pitch_deg) < 10,
           f"roll {att.roll_deg:.1f}, pitch {att.pitch_deg:.1f}, yaw {att.yaw_deg:.1f} deg (level the frame)")

    # ---- external vision
    if a.vision_test:
        pv = await first(drone.telemetry.position_velocity_ned())
        cov = [float("nan")] * 21
        cov[0] = cov[6] = 0.05 ** 2
        for _ in range(60):  # 3 s at 20 Hz, consistent with the current estimate
            msg = VisionPositionEstimate(int(time.time() * 1e6), PositionBody(pv.position.north_m, pv.position.east_m,
                                         pv.position.down_m), AngleBody(0.0, 0.0, math.radians(att.yaw_deg)),
                                         Covariance(cov), 0)
            await drone.mocap.set_vision_position_estimate(msg)
            await asyncio.sleep(0.05)
        h2 = await first(drone.telemetry.health())
        report("vision position estimate accepted", h2.is_local_position_ok,
               "check in QGC/MAVLink console: listener estimator_aid_src_ev_pos (fused: True)")

    # ---- offboard (motors spin!)
    if a.offboard_test:
        if input("Motors WILL spin. Type PROPS OFF to continue: ").strip() != "PROPS OFF":
            report("offboard test", None, "skipped by operator")
        else:
            pv = await first(drone.telemetry.position_velocity_ned())
            sp = PositionNedYaw(pv.position.north_m, pv.position.east_m, pv.position.down_m, att.yaw_deg)
            await drone.offboard.set_position_ned(sp)
            try:
                await drone.action.arm()
                await drone.offboard.start()
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    await drone.offboard.set_position_ned(sp)
                    await asyncio.sleep(0.05)
                mode = await first(drone.telemetry.flight_mode())
                report("offboard engaged with setpoint stream", mode.name == "OFFBOARD", mode.name)
            except Exception as e:  # noqa: BLE001
                report("offboard engaged with setpoint stream", False, str(e))
            finally:
                try:
                    await drone.offboard.stop()
                except Exception:  # noqa: BLE001
                    pass
                await drone.action.disarm()


async def _wait_connected(drone):
    async for s in drone.core.connection_state():
        if s.is_connected:
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="udp://:14540")
    ap.add_argument("--params", default=str(ROOT / "config/px4/gps_denied_flow.params"))
    ap.add_argument("--vision-test", action="store_true")
    ap.add_argument("--offboard-test", action="store_true")
    ap.add_argument("--allow-gps", action="store_true", help="SITL/SIH with GPS: don't fail on a GPS fix")
    a = ap.parse_args()
    print(f"Bench check {a.url}")
    asyncio.run(run(a))
    out = ROOT / "logs" / f"bench_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    fails = sum(r["result"] == "FAIL" for r in results)
    print(f"\n{len(results) - fails}/{len(results)} passed. Report: {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
