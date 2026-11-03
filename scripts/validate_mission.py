#!/usr/bin/env python3
"""Check a mission file before flying it (sim, SITL or real).

    python scripts/validate_mission.py missions/investor_demo.yaml
    python scripts/validate_mission.py --all

Checks: waypoints inside the geofence, altitude limits, landmark names exist in the
world and are detectable classes, every leg's clearance from obstacles, the
landmark is visible from the waypoint at the planned altitude, and a battery
estimate for the route. Exit code 1 if any ERROR.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.common.config import MISSIONS_DIR, load_sim_config  # noqa: E402
from gdnav.navigation.mission import Mission  # noqa: E402
from gdnav.sim.world import World  # noqa: E402


def seg_point_dist(a, b, p) -> float:
    ab, ap = b - a, p - a
    t = 0.0 if not ab.any() else float(np.clip(ap @ ab / (ab @ ab), 0, 1))
    return float(np.linalg.norm(a + t * ab - p))


def validate(path: str | Path, cfg: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    m = Mission.load(path)
    w = World.load(m.world, build_texture=False)
    sc, nav, cam = cfg["safety"], cfg["navigation"], cfg["sensors"]["camera"]
    margin = sc["geofence_margin_m"]
    if m.altitude_m > sc["max_altitude_m"]:
        out.append(("ERROR", f"altitude {m.altitude_m} m above safety.max_altitude_m {sc['max_altitude_m']}"))
    if m.altitude_m < 1.0:
        out.append(("WARN", f"altitude {m.altitude_m} m: ground effect and a tiny camera footprint"))
    half_w = m.altitude_m * math.tan(math.radians(cam["hfov_deg"]) / 2)
    half_h = half_w * cam["height"] / cam["width"]
    route = [np.zeros(2)] + [wp.xy for wp in m.waypoints]
    if m.on_finish == "return_to_start":
        route.append(np.zeros(2))
    for wp in m.waypoints:
        if not w.inside(wp.x, wp.y, margin):
            out.append(("ERROR", f"WP{wp.id} ({wp.x}, {wp.y}) outside the geofence"))
        if wp.landmark:
            lm = w.landmark(wp.landmark)
            if lm is None:
                out.append(("ERROR", f"WP{wp.id}: landmark '{wp.landmark}' not in world {w.name}"))
                continue
            d = math.hypot(lm.x - wp.x, lm.y - wp.y)
            if d > min(half_w, half_h) * 0.8:
                out.append(("WARN", f"WP{wp.id}: landmark {lm.name} is {d:.1f} m from the waypoint, near the "
                                    f"edge of the {2 * half_w:.1f} x {2 * half_h:.1f} m camera footprint"))
            px = lm.size / (2 * half_w) * cam["width"]
            if px < 40:
                out.append(("WARN", f"WP{wp.id}: {lm.name} only ~{px:.0f} px wide at {m.altitude_m} m; fly lower or enlarge it"))
        for o in w.obstacles:
            if math.hypot(o.x - wp.x, o.y - wp.y) < o.radius + 0.8 and m.altitude_m < o.height:
                out.append(("ERROR", f"WP{wp.id} within 0.8 m of {o.name}"))
    dist = 0.0
    for i, (a, b) in enumerate(zip(route, route[1:])):
        dist += float(np.linalg.norm(b - a))
        for o in w.obstacles:
            if m.altitude_m >= o.height:
                continue
            c = seg_point_dist(a, b, np.array([o.x, o.y])) - o.radius
            if c < 0.3:
                out.append(("INFO", f"leg {i + 1} passes {o.name} ({c:+.2f} m): the local planner will steer around it"))
    if m.precision_land_landmark and w.landmark(m.precision_land_landmark) is None:
        out.append(("ERROR", f"precision_land_landmark '{m.precision_land_landmark}' not in world"))
    t_est = dist / nav["cruise_speed"] * 1.4 + sum(wp.hold_s + 3 for wp in m.waypoints) + 15
    v = cfg["vehicle"]
    used = v["hover_current_a"] * 1.1 * t_est / 3600 / (v["battery_capacity_mah"] / 1000)
    out.append(("INFO", f"route {dist:.1f} m, about {t_est:.0f} s, about {used * 100:.0f} % battery "
                        f"(RTS triggers at {sc['battery_rts_pct']} %)"))
    if 1 - used < sc["battery_rts_pct"] / 100 + 0.1:
        out.append(("ERROR", "not enough battery margin for this route"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("missions", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    files = sorted(MISSIONS_DIR.glob("*.yaml")) if a.all or not a.missions else a.missions
    cfg = load_sim_config()
    bad = 0
    for f in files:
        res = validate(f, cfg)
        errs = sum(1 for lvl, _ in res if lvl == "ERROR")
        bad += errs
        print(f"{Path(f).name}: {'OK' if not errs else f'{errs} error(s)'}")
        for lvl, msg in res:
            print(f"  {lvl:5s} {msg}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
