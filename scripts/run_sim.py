#!/usr/bin/env python3
"""Run a mission headless (no browser) - fast, scriptable, CI-friendly.

    python scripts/run_sim.py                                   # investor_demo, as fast as possible
    python scripts/run_sim.py --mission hello_square
    python scripts/run_sim.py --no-correction                   # vision AI off: flow + IMU only, for comparison
    python scripts/run_sim.py --inject camera_covered@40:6 --inject wind_gust@20:10
    python scripts/run_sim.py --record demo.mp4                 # video for slides / social
    python scripts/run_sim.py --realtime                        # pace at 1x wall clock

Fault names: camera_covered, camera_dropout, dim_lights, link_loss,
featureless_floor, wind_gust, low_battery, false_landmark.
Syntax: name@start_s[:duration_s]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.navigation.state_machine import State  # noqa: E402
from gdnav.runner import build_sim_runner  # noqa: E402


def parse_fault(s: str):
    name, _, rest = s.partition("@")
    start, _, dur = rest.partition(":")
    return name, float(start or 0), float(dur) if dur else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="investor_demo")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-correction", action="store_true")
    ap.add_argument("--inject", action="append", default=[], metavar="FAULT@T[:DUR]")
    ap.add_argument("--log", default="auto", help="JSONL log path ('auto' = logs/<mission>_<time>.jsonl, 'none')")
    ap.add_argument("--record", default=None, help="write an MP4 video of the run")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--timeout", type=float, default=600, help="max simulated seconds")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = ap.parse_args()

    log = None if a.log == "none" else a.log
    runner = build_sim_runner(a.mission, a.config, log, not a.no_correction,
                              annotate=bool(a.record), fault_schedule=[parse_fault(f) for f in a.inject],
                              seed=a.seed)
    video = None
    if a.record:
        from gdnav.tools.video import VideoComposer
        video = VideoComposer(runner, a.record)
    if not a.quiet:
        print(f"Mission '{runner.mission.name}' in world '{runner.world.name}' | detector: {runner.cv.detector_name}"
              f" | landmark correction: {'ON' if runner.landmark_correction else 'OFF'}")
    runner.start()
    wall0 = time.time()
    printed = 0
    while runner.sm.state not in (State.LANDED, State.ABORTED) and runner.t < a.timeout:
        runner.tick()
        if video:
            video.write()
        if a.realtime:
            time.sleep(max(0.0, wall0 + runner.t - time.time()))
        if not a.quiet:
            new = min(runner.event_count - printed, len(runner.events))
            for ev in list(runner.events)[len(runner.events) - new:] if new > 0 else []:
                if ev["type"] in ("state", "waypoint", "landed", "fault", "warn", "px4"):
                    print(f"  [{ev['t']:7.2f}s] {ev['msg']}")
            printed = runner.event_count
    summary = runner.finish()
    runner.close()
    if video:
        video.close()
    if a.json:
        print(json.dumps(summary, indent=2))
    elif not a.quiet:
        print("\n=== Summary ===")
        keys = ["waypoints_done", "waypoints_total", "landmarks_found", "reach_error_mean_m", "reach_error_max_m",
                "rts_error_m", "rts_error_pct", "distance_m", "est_error_mean_m", "est_error_max_m",
                "dr_error_max_m", "fixes_accepted", "fixes_rejected", "collisions", "perception_ms_mean",
                "latency_ms_mean", "battery_pct", "flight_time_s", "failsafe_responses"]
        for k in keys:
            if k in summary:
                print(f"  {k:22s} {summary[k]}")
        print(f"  {'wall_time_s':22s} {time.time() - wall0:.1f}")
        if runner.logger.path:
            print(f"\nLog: {runner.logger.path}\nReport: python scripts/make_report.py {runner.logger.path}")
        if a.record:
            print(f"Video: {a.record}")
    return 0 if runner.sm.state == State.LANDED else 1


if __name__ == "__main__":
    sys.exit(main())
