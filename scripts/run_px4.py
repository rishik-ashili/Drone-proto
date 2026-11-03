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
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.common.config import LOGS_DIR, load_sim_config  # noqa: E402
from gdnav.navigation.mission import Mission  # noqa: E402
from gdnav.navigation.state_machine import State  # noqa: E402
from gdnav.runner import MissionRunner  # noqa: E402
from gdnav.sim.world import World  # noqa: E402


def start_control_server(runner, lock, port: int) -> None:
    """Tiny HTTP control API for standby mode (ground laptop -> companion)."""
    actions = {"/start": runner.start, "/land": runner.land_now, "/take_control": runner.take_control,
               "/resume": runner.resume}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/status":
                return self._reply(404, {"error": "unknown"})
            with lock:
                s = runner.snapshot()
            self._reply(200, {k: s[k] for k in ("t", "state", "mode", "armed", "battery", "faults", "metrics")})

        def do_POST(self):
            fn = actions.get(self.path)
            if fn is None:
                return self._reply(404, {"error": "unknown"})
            with lock:
                ok = fn() is not False
                state = runner.sm.state.value
            self._reply(200, {"ok": ok, "state": state})

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="hello_square")
    ap.add_argument("--url", default="udp://:14540")
    ap.add_argument("--camera", default=None, help="OpenCV camera index or video file for perception")
    ap.add_argument("--gz-camera", nargs="?", const="/gdnav/down_camera/image", default=None, metavar="TOPIC",
                    help="use the Gazebo downward camera (./scripts/px4_sitl.sh gdnav)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--standby", action="store_true",
                    help="boot-time service mode: never start on its own, wait for POST /start on --control-port")
    ap.add_argument("--control-port", type=int, default=8081)
    a = ap.parse_args()

    from gdnav.interface.px4_mavsdk import Px4MavsdkVehicle

    frame_source = None
    if a.gz_camera:
        from gdnav.interface.gz_camera import GzCamera
        frame_source = GzCamera(a.gz_camera)
        print(f"Perception input: Gazebo camera {a.gz_camera}")
    elif a.camera is not None:
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
    lock = threading.Lock()
    if a.standby:
        start_control_server(runner, lock, a.control_port)
        print(f"Connected. STANDBY - mission '{mission.name}' starts only on command:\n"
              f"  curl -X POST http://<companion-ip>:{a.control_port}/start   (also /land, /take_control, /resume, GET /status)")
    else:
        print(f"Connected. Mission '{mission.name}', log {log}. Starting in 3 s (Ctrl+C to abort)...")
        time.sleep(3)
        runner.start()
    printed = 0
    try:
        while (a.standby or runner.sm.state not in (State.LANDED, State.ABORTED)) and runner.t < a.timeout:
            t0 = time.monotonic()
            with lock:
                runner.tick()
            new = min(runner.event_count - printed, len(runner.events))
            for ev in list(runner.events)[len(runner.events) - new:] if new > 0 else []:
                print(f"  [{ev['t']:7.2f}s] {ev['msg']}", flush=True)
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
