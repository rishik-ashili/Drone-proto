"""Ground dashboard: live 3D view, onboard camera, state machine, health,
metrics and failure injection, served to any browser on the local network.

Only the Python standard library is used (no web framework to install):
  GET  /                 dashboard page
  GET  /api/world        arena, landmarks, obstacles, mission, fault list
  GET  /api/arena.jpg    floor texture (for the 3D ground and the 2D map)
  GET  /api/stream       Server-Sent Events, 10 Hz state snapshots
  GET  /api/trail?since= trajectory samples (true / estimated / dead-reckoning)
  GET  /api/camera.mjpg  annotated onboard camera (MJPEG)
  POST /api/command      {"cmd": start|pause|resume|reset|speed|inject|clear|mission, ...}
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from gdnav.common.config import MISSIONS_DIR
from gdnav.runner import MissionRunner, build_sim_runner
from gdnav.sim.sim_vehicle import FAULTS

STATIC = Path(__file__).parent / "static"


class SimApp:
    """Owns the simulation thread and the latest snapshot."""

    def __init__(self, mission: str, config: str | None = None, speed: float = 1.0,
                 landmark_correction: bool = True, autostart: bool = False, log: bool = True):
        self.mission_name = mission
        self.config = config
        self.speed = speed
        self.landmark_correction = landmark_correction
        self.log = log
        self.lock = threading.RLock()
        self.paused = False
        self.snapshot_json = b"{}"
        self.version = 0
        self.runner: MissionRunner = self._build()
        self._arena_jpg = self._encode_arena()
        if autostart:
            self.runner.start()
        threading.Thread(target=self._loop, daemon=True, name="sim").start()

    def _build(self) -> MissionRunner:
        return build_sim_runner(self.mission_name, self.config, "auto" if self.log else None,
                                self.landmark_correction, annotate=True)

    def _encode_arena(self) -> bytes:
        tex = self.runner.world.texture
        s = 2048 / max(tex.shape[:2])
        small = cv2.resize(tex, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else tex
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # ---------------------------------------------------------- sim thread
    def _loop(self) -> None:
        next_wall = time.monotonic()
        last_snap = 0.0
        while True:
            try:
                if self.paused:
                    time.sleep(0.05)
                    next_wall = time.monotonic()
                else:
                    with self.lock:
                        self.runner.tick()
                        dt = self.runner.dt
                    next_wall += dt / max(self.speed, 0.05)
                    lag = next_wall - time.monotonic()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.5:  # can't keep up at this speed: don't try to catch up
                        next_wall = time.monotonic()
                now = time.monotonic()
                if now - last_snap >= 0.1:
                    last_snap = now
                    with self.lock:
                        snap = self.runner.snapshot()
                    snap.update(paused=self.paused, speed=self.speed, mission=self.mission_name,
                                correction=self.landmark_correction)
                    self.snapshot_json = json.dumps(snap).encode()
                    self.version += 1
            except Exception:  # keep the dashboard alive, show the error in the event log
                traceback.print_exc()
                time.sleep(0.5)

    # ------------------------------------------------------------ commands
    def command(self, c: dict) -> dict:
        cmd = c.get("cmd")
        with self.lock:
            if cmd == "start":
                self.paused = False
                self.runner.start()
            elif cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "speed":
                self.speed = float(c.get("value", 1.0))
            elif cmd == "inject":
                self.runner.inject(c["fault"], c.get("duration"))
            elif cmd == "clear":
                self.runner.clear(c["fault"])
            elif cmd in ("reset", "mission", "correction"):
                if cmd == "mission":
                    self.mission_name = c["value"]
                if cmd == "correction":
                    self.landmark_correction = bool(c["value"])
                self.runner.close()
                self.runner = self._build()
                self._arena_jpg = self._encode_arena()
                self.paused = False
            else:
                return {"ok": False, "error": f"unknown command {cmd}"}
        return {"ok": True}

    def world_info(self) -> dict:
        r = self.runner
        return {"world": r.world.to_dict(), "mission": r.mission.to_dict(),
                "faults": FAULTS, "missions": sorted(p.stem for p in MISSIONS_DIR.glob("*.yaml")),
                "camera": r.cfg["sensors"]["camera"], "detector": r.cv.detector_name,
                "home": [0.0, 0.0]}


def make_handler(app: SimApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            url = urlparse(self.path)
            p = url.path
            try:
                if p in ("/", "/index.html"):
                    return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if p.startswith("/static/"):
                    f = (STATIC / p[len("/static/"):]).resolve()
                    if STATIC.resolve() not in f.parents or not f.is_file():
                        return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    if f.suffix == ".js":
                        ctype = "text/javascript"
                    return self._send(f.read_bytes(), ctype, cache=f.parent.name == "vendor")
                if p == "/api/world":
                    with app.lock:
                        return self._json(app.world_info())
                if p == "/api/arena.jpg":
                    return self._send(app._arena_jpg, "image/jpeg")
                if p == "/api/state":
                    return self._send(app.snapshot_json, "application/json")
                if p == "/api/trail":
                    since = float(parse_qs(url.query).get("since", ["-1"])[0])
                    with app.lock:
                        trail = app.runner.trail
                        pts = [r for r in trail[-4000:] if r[0] > since] if trail else []
                    return self._json({"points": pts})
                if p == "/api/camera.jpg":
                    jpg = app.runner.jpeg or b""
                    return self._send(jpg, "image/jpeg")
                if p == "/api/stream":
                    return self._stream()
                if p == "/api/camera.mjpg":
                    return self._mjpeg()
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if urlparse(self.path).path != "/api/command":
                return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            n = int(self.headers.get("Content-Length", 0))
            try:
                res = app.command(json.loads(self.rfile.read(n) or b"{}"))
                self._json(res, HTTPStatus.OK if res.get("ok") else HTTPStatus.BAD_REQUEST)
            except Exception as e:  # report, don't crash the server
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)

        def _stream(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            while True:
                if app.version != seen:
                    seen = app.version
                    self.wfile.write(b"data: " + app.snapshot_json + b"\n\n")
                    self.wfile.flush()
                time.sleep(0.05)

        def _mjpeg(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            while True:
                jpg = app.runner.jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.03)

    return Handler


def serve(app: SimApp, host: str = "127.0.0.1", port: int = 8080) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"\n  Dashboard running:  http://{shown}:{port}\n  (Ctrl+C to stop)\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.runner.close()
        httpd.server_close()
