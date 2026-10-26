#!/usr/bin/env python3
"""Replay a recorded flight in the dashboard.

    python scripts/replay_log.py logs/<file>.jsonl --speed 4
Then open http://localhost:8080 and press Start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.dashboard.server import SimApp, serve  # noqa: E402
from gdnav.tools.replay import ReplayRunner  # noqa: E402


class ReplayApp(SimApp):
    def __init__(self, log: str, speed: float):
        self.log_file = log
        super().__init__(mission="replay", speed=speed, log=False)

    def _build(self):
        return ReplayRunner(self.log_file)

    def world_info(self) -> dict:
        info = super().world_info()
        info["missions"] = [self.runner.mission.name]
        return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    serve(ReplayApp(a.log, a.speed), a.host, a.port)


if __name__ == "__main__":
    main()
