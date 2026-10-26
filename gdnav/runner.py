"""MissionRunner: the companion-computer main loop, wired to a vehicle.

One ``tick()`` = one navigation cycle (20 Hz by default):
  camera frame -> cv_node -> landmark fixes -> estimator (via the vehicle)
  -> health_monitor -> mission state machine -> Offboard setpoint
plus logging and metrics. The same loop drives the built-in simulator or a
real PX4 (SITL / Pixhawk) through the VehicleInterface.
"""
from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from gdnav.common.config import LOGS_DIR, load_sim_config
from gdnav.common.logger import FlightLogger
from gdnav.interface.vehicle import VehicleInterface
from gdnav.navigation.geofence import Geofence
from gdnav.navigation.local_planner import LocalPlanner
from gdnav.navigation.mission import Mission
from gdnav.navigation.state_machine import FLYING_STATES, VISION_STATES, MissionStateMachine, State
from gdnav.perception.cv_node import CvNode
from gdnav.safety.health_monitor import HealthMonitor
from gdnav.sim.sim_vehicle import FAULTS, SimVehicle
from gdnav.sim.world import World


class MissionRunner:
    def __init__(self, cfg: dict, mission: Mission, world: World, vehicle: VehicleInterface,
                 log_path: str | Path | None = None, landmark_correction: bool = True,
                 annotate: bool = True, fault_schedule: list[tuple[str, float, float]] | None = None):
        self.cfg = cfg
        self.mission = mission
        self.world = world
        self.vehicle = vehicle
        self.is_sim = isinstance(vehicle, SimVehicle)
        self.landmark_correction = landmark_correction
        self.annotate = annotate
        self.dt = 1.0 / cfg["rates_hz"]["nav"]
        self.cv = CvNode(cfg["perception"], world, cfg["sensors"]["camera"])
        self.health = HealthMonitor(cfg["safety"])
        self.planner = LocalPlanner(cfg["navigation"]["avoidance"])
        sc = cfg["safety"]
        self.fence = Geofence(world.x_min, world.x_max, world.y_min, world.y_max,
                              sc["max_altitude_m"], sc["geofence_margin_m"])
        self.events: deque[dict] = deque(maxlen=200)
        self.event_count = 0
        self.sm = MissionStateMachine(cfg["navigation"], sc, mission, vehicle, self.planner, self.fence,
                                      self.health, self._event, self._truth)
        self.logger = FlightLogger(log_path)
        self.logger.write("meta", mission=mission.to_dict(), world=world.to_dict(),
                          landmark_correction=landmark_correction, detector=self.cv.detector_name,
                          backend=type(vehicle).__name__)
        self.fault_schedule = sorted(fault_schedule or [], key=lambda f: f[1])
        self.active_injections: dict[str, float] = {}
        self._pending_response: dict[str, float] = {}
        self.failsafe_responses: list[dict] = []

        self.t = 0.0
        self._last_frame_t = -1.0
        self.last_perception = None
        self.jpeg: bytes | None = None
        self.trail: list[list[float]] = []
        self._trail_next = 0.0
        self.tick_count = 0
        self._tick_walls: deque[float] = deque(maxlen=100)
        self.err_stats = {"est_sum": 0.0, "est_max": 0.0, "dr_max": 0.0, "n": 0, "dr_now": 0.0, "est_now": 0.0}
        self.fix_stats = {"accepted": 0, "rejected": 0, "sent": 0}
        self.perc_ms: deque[float] = deque(maxlen=60)
        self.latency_ms: deque[float] = deque(maxlen=60)
        self.summary: dict | None = None

    # ----------------------------------------------------------- utilities
    def _now(self) -> float:
        return self.vehicle.now()

    def _truth(self):
        return self.vehicle.truth() if self.is_sim else None

    def _event(self, kind: str, message: str, data: dict) -> None:
        ev = {"t": round(self.t, 2), "type": kind, "msg": message}
        self.events.append(ev)
        self.event_count += 1
        self.logger.write("event", t=self.t, type=kind, msg=message, data=data)
        if kind == "state" and self._pending_response:
            to = data.get("to")
            if to in (State.FAILSAFE_HOLD.value, State.LAND.value, State.RETURN_TO_START.value):
                for name, t_inj in list(self._pending_response.items()):
                    resp = {"fault": name, "response": to, "response_time_s": round(self.t - t_inj, 2)}
                    self.failsafe_responses.append(resp)
                    self.logger.write("event", t=self.t, type="failsafe_response", msg=str(resp), data=resp)
                    del self._pending_response[name]

    # ------------------------------------------------------------ commands
    def start(self) -> None:
        self.sm.start()

    def inject(self, name: str, duration: float | None = None) -> None:
        if name == "false_landmark":
            self.cv.inject_false = True
        elif self.is_sim:
            self.vehicle.inject(name)
        else:
            raise ValueError("Fault injection needs the simulated vehicle")
        self.active_injections[name] = self.t + duration if duration else float("inf")
        if name in ("camera_covered", "camera_dropout", "link_loss", "featureless_floor", "low_battery"):
            self._pending_response[name] = self.t
        self._event("fault", f"Injected: {FAULTS.get(name, name)}", {"fault": name})

    def clear(self, name: str) -> None:
        if name == "false_landmark":
            self.cv.inject_false = False
        elif self.is_sim:
            self.vehicle.clear(name)
        self.active_injections.pop(name, None)
        self._pending_response.pop(name, None)
        self._event("fault", f"Cleared: {FAULTS.get(name, name)}", {"fault": name})

    # ---------------------------------------------------------------- tick
    def tick(self) -> None:
        w0 = time.perf_counter()
        if self.is_sim:
            self.vehicle.advance(self.dt)
        self.t = now = self._now()
        for name, t0, dur in [f for f in self.fault_schedule if f[1] <= now]:
            self.inject(name, dur)
        self.fault_schedule = [f for f in self.fault_schedule if f[1] > now]
        for name, t_end in list(self.active_injections.items()):
            if now >= t_end:
                self.clear(name)
        if self.is_sim:
            for lvl, msg in self.vehicle.fc.events:
                self._event("px4", msg, {"level": lvl})
            self.vehicle.fc.events.clear()

        telem = self.vehicle.telemetry()

        # ---- perception
        frame = self.vehicle.latest_frame()
        new_frame_t = None
        perc = None
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            new_frame_t = frame.t
            perc = self.cv.process(frame, telem.pos, annotate=self.annotate)
            self.last_perception = perc
            self.perc_ms.append(perc.proc_ms)
            for o in perc.observations:
                if o.confirmed and self.landmark_correction:
                    o.accepted = self.vehicle.send_vision_position(frame.t, o.fix_xy, o.std, o.yaw_fix)
                    self.fix_stats["sent"] += 1
                    self.fix_stats["accepted" if o.accepted else "rejected"] += 1
            self.sm.observe(now, perc.observations)
            if self.annotate and perc.annotated is not None:
                ok, buf = cv2.imencode(".jpg", perc.annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpeg = buf.tobytes()

        # ---- safety
        flying = self.sm.state in FLYING_STATES
        img_stats = (perc.image_mean, perc.image_std) if perc else None
        vision_needed = self.sm.state in VISION_STATES or (
            self.sm.state == State.FAILSAFE_HOLD and self.sm.resume_state in VISION_STATES)
        self.health.evaluate(now, telem, new_frame_t, img_stats, flying, vision_needed,
                             not self.fence.breached(telem.pos))

        # ---- navigation
        self.sm.step(now, telem, self.vehicle.depth_scan())
        if perc is not None:
            # capture -> setpoint latency (perception time + camera latency)
            self.latency_ms.append(perc.proc_ms + 1000 * self.cfg["sensors"]["camera"]["latency_s"]
                                   + 1000 * max(0.0, now - perc.t))

        self._metrics(telem)
        self._log(telem, perc)
        self.tick_count += 1
        self._tick_walls.append(time.perf_counter() - w0)
        if self.sm.state == State.LANDED and self.summary is None:
            self.finish()

    # --------------------------------------------------------- bookkeeping
    def _metrics(self, telem) -> None:
        tr = self._truth()
        if tr is None:
            return
        est_err = float(np.linalg.norm(tr["pos"][:2] - telem.pos[:2]))
        dr_err = float(np.linalg.norm(tr["pos"][:2] - tr["dr_pos"][:2]))
        s = self.err_stats
        s["est_now"], s["dr_now"] = est_err, dr_err
        if self.sm.state in FLYING_STATES:
            s["est_sum"] += est_err
            s["n"] += 1
            s["est_max"] = max(s["est_max"], est_err)
            s["dr_max"] = max(s["dr_max"], dr_err)
        if self.t >= self._trail_next:
            self._trail_next = self.t + 0.1
            p, e, d = tr["pos"], telem.pos, tr["dr_pos"]
            self.trail.append([round(self.t, 2), *(round(float(v), 3) for v in (p[0], p[1], -p[2], e[0], e[1], -e[2], d[0], d[1])),
                               round(est_err, 3), round(dr_err, 3)])

    def _log(self, telem, perc) -> None:
        tr = self._truth()
        sp = self.sm.last_sp
        rec = {"t": round(self.t, 3), "state": self.sm.state.value, "mode": telem.mode,
               "est_pos": telem.pos, "est_vel": telem.vel, "est_att": telem.att, "pos_std": telem.pos_std,
               "battery": round(telem.battery, 4), "flow_q": telem.flow_quality,
               "sp": None if sp is None else {"kind": sp.kind, "pos": sp.pos, "vel": sp.vel, "yaw": sp.yaw},
               "faults": [f.name for f in self.health.faults.values()]}
        if tr is not None:
            rec.update(true_pos=tr["pos"], true_att=tr["att"], dr_pos=tr["dr_pos"])
        if perc is not None:
            rec["det"] = [{"cls": d.cls, "src": d.source, "conf": round(d.conf, 3), "tag": d.tag_id,
                           "box": [round(v, 1) for v in d.bbox]} for d in perc.detections]
            rec["obs"] = [{"name": o.name, "off": o.offset_ned, "fix": o.fix_xy, "std": round(o.std, 3),
                           "conf": o.confirmed, "acc": o.accepted} for o in perc.observations]
            rec["perc_ms"] = round(perc.proc_ms, 2)
        self.logger.write("tick", **rec)

    def metrics(self) -> dict:
        res = self.sm.mgr.results
        reach = [r["reach_error_m"] for r in res if "reach_error_m" in r]
        s = self.err_stats
        tw = list(self._tick_walls)
        m = {
            "waypoints_done": len(res),
            "waypoints_total": len(self.mission.waypoints),
            "landmarks_found": sum(1 for r in res if r.get("found")),
            "reach_error_mean_m": round(float(np.mean(reach)), 3) if reach else None,
            "reach_error_max_m": round(float(np.max(reach)), 3) if reach else None,
            "est_error_now_m": round(s["est_now"], 3),
            "est_error_mean_m": round(s["est_sum"] / s["n"], 3) if s["n"] else None,
            "est_error_max_m": round(s["est_max"], 3),
            "dr_error_now_m": round(s["dr_now"], 3),
            "dr_error_max_m": round(s["dr_max"], 3),
            "fixes_accepted": self.fix_stats["accepted"],
            "fixes_rejected": self.fix_stats["rejected"],
            "perception_ms_mean": round(float(np.mean(self.perc_ms)), 1) if self.perc_ms else None,
            "perception_fps": round(self.last_perception.stats.get("fps", 0.0), 1) if self.last_perception else None,
            "latency_ms_mean": round(float(np.mean(self.latency_ms)), 1) if self.latency_ms else None,
            "nav_rate_hz": self.cfg["rates_hz"]["nav"],
            "nav_tick_ms_mean": round(1000 * float(np.mean(tw)), 2) if tw else None,
            "nav_tick_ms_max": round(1000 * float(np.max(tw)), 2) if tw else None,
            "failsafe_responses": list(self.failsafe_responses),
            "flight_time_s": round(self.t, 1),
            "detector": self.cv.detector_name,
            "landmark_correction": self.landmark_correction,
        }
        if self.sm.rts_result:
            m.update(self.sm.rts_result)
        if self.is_sim:
            tr = self.vehicle.truth()
            m["distance_m"] = round(tr["distance"], 2)
            m["battery_pct"] = round(self.vehicle.quad.battery.remaining * 100, 1)
            m["collisions"] = len(self.vehicle.collisions)
        return m

    def finish(self) -> dict:
        self.summary = self.metrics()
        self.summary["waypoints"] = self.sm.mgr.results
        self.logger.write("summary", t=self.t, **self.summary)
        self.logger.flush()
        return self.summary

    def close(self) -> None:
        if self.summary is None:
            self.finish()
        self.logger.close()

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict:
        telem = self.vehicle.telemetry()
        tr = self._truth()
        perc = self.last_perception
        sp = self.sm.last_sp
        snap = {
            "t": round(self.t, 2),
            "state": self.sm.state.value,
            "mode": telem.mode,
            "armed": telem.armed,
            "est": {"pos": telem.pos.round(3).tolist(), "att": telem.att.round(4).tolist(),
                    "vel": telem.vel.round(3).tolist(), "std": round(telem.pos_std, 3)},
            "battery": round(telem.battery * 100, 1),
            "voltage": round(telem.voltage, 2),
            "flow_quality": telem.flow_quality,
            "range": round(telem.range_m, 2),
            "setpoint": None if sp is None else {"kind": sp.kind, "pos": sp.pos.round(2).tolist(),
                                                 "vel": sp.vel.round(2).tolist()},
            "wp_index": self.sm.mgr.index,
            "wp_results": self.sm.mgr.results,
            "faults": [{"name": f.name, "action": f.action, "msg": f.message} for f in self.health.faults.values()],
            "injections": list(self.active_injections),
            "planner": {"active": self.planner.active,
                        "closest": None if math.isinf(self.planner.closest) else round(self.planner.closest, 2)},
            "perception": None if perc is None else {
                "fps": round(perc.stats.get("fps", 0), 1), "ms": round(perc.proc_ms, 1),
                "detector": perc.stats.get("detector"),
                "detections": [{"cls": d.cls, "src": d.source, "conf": round(d.conf, 2), "tag": d.tag_id}
                               for d in perc.detections],
                "observations": [{"name": o.name, "src": o.source, "confirmed": o.confirmed,
                                  "accepted": o.accepted, "std": round(o.std, 3),
                                  "off": [round(float(v), 2) for v in o.offset_ned]} for o in perc.observations]},
            "metrics": self.metrics(),
            "events": list(self.events)[-30:],
            "mission_complete": self.sm.mission_complete,
            "depth": None,
        }
        if tr is not None:
            snap["true"] = {"pos": tr["pos"].round(3).tolist(), "att": tr["att"].round(4).tolist(),
                            "wind": tr["wind"].round(2).tolist()}
            snap["dr"] = {"pos": tr["dr_pos"].round(3).tolist()}
        scan = self.vehicle.depth_scan()
        if scan is not None:
            snap["depth"] = {"angles": [round(float(a), 3) for a in scan[0]],
                             "ranges": [round(float(r), 2) for r in scan[1]]}
        return snap


def build_sim_runner(mission_name: str = "investor_demo", config_path: str | None = None,
                     log_path: str | Path | None = "auto", landmark_correction: bool = True,
                     annotate: bool = True, fault_schedule=None, overrides: dict | None = None,
                     seed: int | None = None) -> MissionRunner:
    cfg = load_sim_config(config_path, overrides)
    if seed is not None:
        cfg["seed"] = seed
    mission = Mission.load(mission_name)
    world = World.load(mission.world)
    vehicle = SimVehicle(cfg, world)
    if log_path == "auto":
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / f"{mission.name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    return MissionRunner(cfg, mission, world, vehicle, log_path, landmark_correction, annotate, fault_schedule)
