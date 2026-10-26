#!/usr/bin/env python3
"""Fly a mission on real PX4 - SITL (Gazebo) or a Pixhawk - with the same
mission code as the built-in simulator.

  # 1. start PX4 SITL with the optical-flow quadcopter (see scripts/px4_sitl.sh)
  # 2. then:
  python scripts/run_px4.py --mission hello_square
  python scripts/run_px4.py --mission investor_demo --camera 0      # USB camera as perception input
  python scripts/run_px4.py --url serial:///dev/ttyACM0:921600       # real Pixhawk (props OFF first!)

SAFETY: on real hardware run props-off bench tests first, then tethered, with an
RC transmitter and kill switch in the hands of a pilot (plan, section 12).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.common.config import LOGS_DIR, load_sim_config  # noqa: E402
from gdnav.navigation.mission import Mission  # noqa: E402
from gdnav.navigation.state_machine import State  # noqa: E402
from gdnav.runner import MissionRunner  # noqa: E402
from gdnav.sim.world import World  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="hello_square")
    ap.add_argument("--url", default="udp://:14540")
    ap.add_argument("--camera", default=None, help="OpenCV camera index or video file for perception")
    ap.add_argument("--config", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    from gdnav.interface.px4_mavsdk import Px4MavsdkVehicle

    frame_source = None
    if a.camera is not None:
        import cv2
        cap = cv2.VideoCapture(int(a.camera) if a.camera.isdigit() else a.camera)

        def read_camera():
            ok, img = cap.read()
            return cv2.resize(img, (640, 480)) if ok else None

        frame_source = read_camera

    cfg = load_sim_config(a.config)
    mission = Mission.load(a.mission)
    world = World.load(mission.world, build_texture=False)  # landmark map only
    vehicle = Px4MavsdkVehicle(a.url, frame_source)
    LOGS_DIR.mkdir(exist_ok=True)
    log = LOGS_DIR / f"px4_{mission.name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    runner = MissionRunner(cfg, mission, world, vehicle, log, landmark_correction=True, annotate=False)
    print(f"Connected. Mission '{mission.name}', log {log}. Starting in 3 s (Ctrl+C to abort)...")
    time.sleep(3)
    runner.start()
    printed = 0
    try:
        while runner.sm.state not in (State.LANDED, State.ABORTED) and runner.t < a.timeout:
            t0 = time.monotonic()
            runner.tick()
            new = min(runner.event_count - printed, len(runner.events))
            for ev in list(runner.events)[len(runner.events) - new:] if new > 0 else []:
                print(f"  [{ev['t']:7.2f}s] {ev['msg']}")
            printed = runner.event_count
            time.sleep(max(0.0, runner.dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("Interrupted - commanding LAND")
        vehicle.land()
    summary = runner.finish()
    runner.close()
    vehicle.close()
    print({k: v for k, v in summary.items() if k != "waypoints"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
