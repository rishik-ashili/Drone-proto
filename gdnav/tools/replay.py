"""Replay a recorded flight log into the dashboard (no simulation, no camera).

ReplayRunner exposes the subset of the MissionRunner API the dashboard uses,
driven by the 'tick' records of a JSONL log, so any flight (simulated, SITL or
real, as long as it used the common log format) can be reviewed later.
"""
from __future__ import annotations

import numpy as np

from gdnav.common.config import load_sim_config
from gdnav.common.logger import read_log
from gdnav.navigation.mission import Mission, Waypoint
from gdnav.sim.world import World


class _Cv:
    def __init__(self, name):
        self.detector_name = name


class ReplayRunner:
    def __init__(self, log_path):
        d = read_log(log_path)
        self.meta = d["meta"][0]
        self.ticks = d["tick"]
        self.events_all = d["event"]
        self.summary = d["summary"][-1] if d["summary"] else {}
        self.cfg = load_sim_config()
        self.dt = 1.0 / self.cfg["rates_hz"]["nav"]
        self.world = World.load(self.meta["world"]["name"])
        m = self.meta["mission"]
        self.mission = Mission(name=m["name"], world=self.world.name, altitude_m=m["altitude_m"],
                               waypoints=[Waypoint(**w) for w in m["waypoints"]], on_finish=m["on_finish"],
                               precision_land_landmark=m.get("precision_land_landmark"))
        self.cv = _Cv(f"replay of {self.meta.get('detector')}")
        self.jpeg = None
        self.i = 0
        self.playing = False
        self.trail: list[list[float]] = []
        self.t = 0.0

    def start(self):
        self.playing = True

    def inject(self, *a, **k):
        raise ValueError("fault injection is not available in replay mode")

    clear = inject

    def close(self):
        pass

    def tick(self):
        if not self.playing or self.i >= len(self.ticks) - 1:
            return
        self.i += 1
        k = self.ticks[self.i]
        self.t = k["t"]
        if "true_pos" in k and (not self.trail or self.t - self.trail[-1][0] >= 0.1):
            tp, ep, dp = k["true_pos"], k["est_pos"], k["dr_pos"]
            e = float(np.hypot(tp[0] - ep[0], tp[1] - ep[1]))
            dr = float(np.hypot(tp[0] - dp[0], tp[1] - dp[1]))
            self.trail.append([self.t, tp[0], tp[1], -tp[2], ep[0], ep[1], -ep[2], dp[0], dp[1], e, dr])

    def snapshot(self) -> dict:
        k = self.ticks[self.i]
        evs = [{"t": e["t"], "type": e["type"], "msg": e["msg"]} for e in self.events_all if e["t"] <= k["t"]]
        wps = [e["data"] for e in self.events_all if e["type"] == "waypoint" and e["t"] <= k["t"]]
        last = self.trail[-1] if self.trail else [0] * 11
        done = self.i >= len(self.ticks) - 1
        metrics = {"est_error_now_m": last[9], "dr_error_now_m": last[10],
                   "est_error_mean_m": None, "est_error_max_m": max((p[9] for p in self.trail), default=0),
                   "dr_error_max_m": max((p[10] for p in self.trail), default=0),
                   "reach_error_mean_m": round(float(np.mean([w["reach_error_m"] for w in wps])), 3)
                   if wps and "reach_error_m" in wps[0] else None,
                   "reach_error_max_m": None, "waypoints_done": len(wps),
                   "waypoints_total": len(self.mission.waypoints),
                   "rts_error_m": self.summary.get("rts_error_m") if done else None,
                   "rts_error_pct": self.summary.get("rts_error_pct") if done else None,
                   "distance_m": self.summary.get("distance_m") if done else None,
                   "battery_pct": k["battery"] * 100, "fixes_accepted": 0, "fixes_rejected": 0,
                   "latency_ms_mean": None}
        snap = {"t": k["t"], "state": k["state"], "mode": k["mode"], "armed": k["mode"] != "DISARMED",
                "est": {"pos": k["est_pos"], "att": k["est_att"], "vel": k["est_vel"], "std": k["pos_std"]},
                "battery": k["battery"] * 100, "voltage": 0.0, "flow_quality": k["flow_q"], "range": -k["est_pos"][2],
                "setpoint": k.get("sp"), "wp_index": len(wps), "wp_results": wps,
                "faults": [{"name": f, "action": "hold", "msg": f} for f in k["faults"]], "injections": [],
                "planner": {"active": False, "closest": None}, "perception": None, "metrics": metrics,
                "events": evs[-30:], "mission_complete": done, "depth": None}
        if "true_pos" in k:
            snap["true"] = {"pos": k["true_pos"], "att": k["true_att"], "wind": [0, 0]}
            snap["dr"] = {"pos": k["dr_pos"]}
        return snap
