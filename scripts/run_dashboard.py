#!/usr/bin/env python3
"""Start the simulator with the live investor dashboard.

    python scripts/run_dashboard.py                       # http://localhost:8080
    python scripts/run_dashboard.py --mission hello_square --speed 2
    python scripts/run_dashboard.py --host 0.0.0.0        # show it on another screen / tablet
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.dashboard.server import SimApp, serve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="investor_demo")
    ap.add_argument("--config", default=None, help="YAML overriding config/sim.yaml")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed vs real time")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--autostart", action="store_true", help="start the mission immediately")
    ap.add_argument("--no-correction", action="store_true", help="disable landmark fixes (dead reckoning)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    app = SimApp(a.mission, a.config, a.speed, not a.no_correction, a.autostart, log=not a.no_log)
    if a.open:  # give the server a moment to start listening
        threading.Timer(1.5, webbrowser.open, args=(f"http://localhost:{a.port}",)).start()
    serve(app, a.host, a.port)


if __name__ == "__main__":
    main()
