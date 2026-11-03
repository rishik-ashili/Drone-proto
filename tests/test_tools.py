"""Command-line tools that need no hardware."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_validate_mission_all_ok():
    r = run("scripts/validate_mission.py", "--all")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "investor_demo.yaml: OK" in r.stdout


def test_validate_mission_catches_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nworld: indoor_arena\naltitude_m: 9\nwaypoints:\n"
                   "  - {id: 1, x: 40.0, y: 0.0, landmark: nope}\non_finish: land\n")
    r = run("scripts/validate_mission.py", str(bad))
    assert r.returncode == 1 and "outside the geofence" in r.stdout and "not in world" in r.stdout


def test_printables(tmp_path):
    r = run("scripts/make_printables.py", "--only", "tag_07", "--size-m", "0.3", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("*.pdf"))


def test_export_gazebo(tmp_path):
    import xml.dom.minidom
    r = run("scripts/export_gazebo.py", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    xml.dom.minidom.parse(str(tmp_path / "worlds" / "gdnav_indoor_arena.sdf"))
    sdf = (tmp_path / "models" / "x500_flow_cam_down" / "model.sdf").read_text()
    assert "model://x500_flow" in sdf and "/gdnav/down_camera/image" in sdf


def test_standby_control_server_starts_mission_only_on_command():
    import json
    import socket
    import threading
    import urllib.request

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_px4
    from gdnav.runner import build_sim_runner

    r = build_sim_runner("hello_square", log_path=None, annotate=False)
    lock = threading.Lock()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    run_px4.start_control_server(r, lock, port)
    for _ in range(40):
        r.tick()
    assert r.sm.state.value == "IDLE"          # standby: nothing happens by itself
    req = urllib.request.Request(f"http://127.0.0.1:{port}/start", method="POST", data=b"")
    assert json.load(urllib.request.urlopen(req))["ok"]
    for _ in range(40):
        r.tick()
    status = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/status"))
    assert status["state"] in ("TAKEOFF", "GOTO_WAYPOINT")
