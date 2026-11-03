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

from gdnav.bridge.landmark_vio_fusion import LandmarkVioFusion
from gdnav.common.config import LOGS_DIR, REPO_ROOT, load_sim_config
from gdnav.common.logger import FlightLogger
from gdnav.interface.vehicle import VehicleInterface
from gdnav.navigation.geofence import Geofence
from gdnav.navigation.local_planner import LocalPlanner
from gdnav.navigation.mission import Mission
from gdnav.navigation.state_machine import FLYING_STATES, VISION_STATES, MissionStateMachine, State
from gdnav.perception.cv_node import CvNode
from gdnav.perception.map_matcher import FloorMap, MapMatcher
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
                                      self.health, self._event, self._truth, vision=landmark_correction)
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

        # Phase 4: teach-and-repeat floor-map matching
        mc = cfg.get("map_matching", {})
        self.map_cfg = mc
        self.map_matcher: MapMatcher | None = None
        self.floor_map: FloorMap | None = None
        self.teach_map: FloorMap | None = None
        self.map_stats = {"fixes": 0, "rejected": 0, "tries": 0, "last_shift": None, "taught": 0, "ms": 0.0}
        self._map_next = 0.0
        self._last_fix_t = -1e9
        if mc.get("map") or mc.get("teach_out"):
            self.map_matcher = MapMatcher(self.cv.K, mc["res"], mc["patch_m"], mc["search_m"], mc["min_inliers"])
            if mc.get("map"):
                p = Path(mc["map"])
                self.floor_map = FloorMap.load(p if p.is_absolute() else REPO_ROOT / p)
            if mc.get("teach_out"):
                self.teach_map = FloorMap(world.x_min, world.x_max, world.y_min, world.y_max, mc["res"])

        # Phase 3: stereo VIO as the primary navigation source
        self.nav_source = cfg.get("nav_source", "flow")
        self.fusion: LandmarkVioFusion | None = None
        self.vio_state = {"last_t": -1.0, "quality": 0, "samples": 0, "reset": 0, "ev_resets": 0}
        if self.nav_source == "vio":
            tl = vehicle.telemetry()
            vc = cfg["vio"]
            self.fusion = LandmarkVioFusion(vc["fusion_drift_per_m"], math.radians(vc["fusion_yaw_drift_deg_per_m"]),
                                            init=(float(tl.pos[0]), float(tl.pos[1]), float(tl.att[2])))

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

    # operator / pilot commands (dashboard)
    def take_control(self) -> bool:
        return self.sm.take_control()

    def resume(self) -> bool:
        return self.sm.resume()

    def goto(self, xy) -> bool:
        return self.sm.goto_point(xy)

    def land_now(self) -> None:
        self.sm.land_now()

    def sticks(self, values) -> None:
        self.vehicle.manual_control(values)

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
        if self.fusion is not None:
            self._forward_vio(telem)

        # ---- perception
        frame = self.vehicle.latest_frame()
        new_frame_t = None
        perc = None
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            new_frame_t = frame.t
            est_at_exposure = frame.pos if frame.pos is not None else telem.pos
            # a fix describes the drone at exposure time: apply it as a correction of that estimate
            lag = telem.pos[:2] - est_at_exposure[:2]
            perc = self.cv.process(frame, est_at_exposure, annotate=self.annotate)
            self.last_perception = perc
            self.perc_ms.append(perc.proc_ms)
            for o in perc.observations:
                if o.confirmed and self.landmark_correction:
                    o.accepted = self._send_fix(frame.t, o.fix_xy + lag, o.std, o.yaw_fix)
                    if o.accepted:
                        self._last_fix_t = now
                    self.fix_stats["sent"] += 1
                    self.fix_stats["accepted" if o.accepted else "rejected"] += 1
            self.sm.observe(now, perc.observations)
            if self.map_matcher is not None and frame.t >= self._map_next:
                self._map_next = frame.t + 1.0 / self.map_cfg["rate_hz"]
                self._floor_map_step(frame, perc.image, est_at_exposure, lag, telem, now)
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
        if self.fusion is not None:
            lost = flying and now - self.vio_state["last_t"] > 0.5
            self.health.report("vio", lost, "warn", "VIO tracking lost - optical flow fallback", now)

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

    # ------------------------------------------------------------------ VIO
    def _forward_vio(self, telem) -> None:
        """Correct every VIO sample with the landmark transform and stream it to the
        flight controller as external-vision odometry (EKF2_EV_CTRL pos + vel + yaw)."""
        f, vs = self.fusion, self.vio_state
        for s in self.vehicle.vio_samples():
            if s.reset_counter != vs["reset"]:
                self._event("warn", f"VIO re-initialised (reset {s.reset_counter})", {})
            p, yaw = f.on_odometry(s.pos, s.yaw, s.reset_counter,
                                   anchor=(telem.pos, float(telem.att[2]), telem.pos_std))
            v = f.vel_to_map(s.vel)
            std = math.hypot(s.pos_std, f.correction_std)
            self.vehicle.send_odometry(s.t, p[:2], yaw, v[:2], std, s.yaw_std, 0.05,
                                       s.reset_counter + vs["ev_resets"])
            vs.update(last_t=s.t, quality=s.quality, reset=s.reset_counter)
            vs["samples"] += 1

    def _send_fix(self, t: float, xy: np.ndarray, std: float, yaw: float | None) -> bool:
        """Absolute position (and heading) fix -> estimator, directly (flow) or via the VIO correction."""
        if self.fusion is not None:
            return self._fuse_fix_with_vio(xy, std, yaw)
        return self.vehicle.send_vision_position(t, xy, std, yaw)

    def _floor_map_step(self, frame, image, est_at, lag, telem, now) -> None:
        mm, ms = self.map_matcher, self.map_stats
        pos = np.array(est_at, float).copy()
        pos[2] = -self.cv.localizer.height(frame.range_m, frame.att)
        if -pos[2] < 0.8 or self.sm.state not in FLYING_STATES:
            return
        if self.teach_map is not None and telem.pos_std < self.map_cfg["teach_max_std"] and now - self._last_fix_t < 3.0:
            mm.teach(self.teach_map, image, pos, frame.att)
            ms["taught"] += 1
        if self.floor_map is not None:
            ms["tries"] += 1
            fix = mm.match(self.floor_map, image, pos, frame.att, float(frame.att[2]))
            ms["ms"] = mm.last_ms
            if fix is None:
                return
            yaw_now = fix.yaw + float(telem.att[2] - frame.att[2])
            if self._send_fix(frame.t, fix.xy + lag, fix.std, yaw_now):
                ms["fixes"] += 1
                ms["last_shift"] = round(fix.shift_m, 3)
            else:
                ms["rejected"] += 1

    def _fuse_fix_with_vio(self, xy, std, yaw) -> bool:
        f = self.fusion
        if f.last_odom is None:
            return False
        before = f.last_corrected[0][:2].copy()
        ok = f.on_landmark_fix(xy, std, yaw_fix=yaw, yaw_odom=f.last_odom[1])
        if ok and np.linalg.norm(f.last_corrected[0][:2] - before) > 0.15:
            # a visible jump in the corrected pose: tell EKF2 via the odometry reset counter
            self.vio_state["ev_resets"] += 1
        return ok

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
            "false_positives_per_min": round(self.cv.unassociated / max(self.t / 60.0, 1e-6), 2) if self.t > 1 else None,
            "detector": self.cv.detector_name,
            "landmark_correction": self.landmark_correction,
            "nav_source": self.nav_source,
        }
        if self.map_matcher is not None:
            m.update(map_fixes=self.map_stats["fixes"], map_rejected=self.map_stats["rejected"],
                     map_tries=self.map_stats["tries"])
        if self.fusion is not None:
            m.update(vio_resets=self.vio_state["reset"], vio_samples=self.vio_state["samples"],
                     vio_corrections=self.vio_state["ev_resets"])
        if self.sm.rts_result:
            m.update(self.sm.rts_result)
        if self.is_sim:
            tr = self.vehicle.truth()
            m["distance_m"] = round(tr["distance"], 2)
            m["battery_pct"] = round(self.vehicle.quad.battery.remaining * 100, 1)
            m["collisions"] = len(self.vehicle.collisions)
        return m

    def finish(self) -> dict:
        if self.teach_map is not None and self.map_stats["taught"]:
            out = Path(self.map_cfg["teach_out"])
            out = out if out.is_absolute() else REPO_ROOT / out
            self.teach_map.save(out)
            self._event("info", f"Floor map saved: {out.name} ({self.teach_map.coverage * 100:.0f} % of arena)", {})
            self.map_stats["taught"] = 0
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
            "user_target": None if self.sm.user_target is None else [round(float(v), 2) for v in self.sm.user_target],
            "nav_source": self.nav_source,
            "floor_map": None if self.map_matcher is None else {
                "mode": "repeat" if self.floor_map is not None else "teach", **self.map_stats},
            "vio": None if self.fusion is None else {
                "tracking": self.t - self.vio_state["last_t"] < 0.5, "quality": self.vio_state["quality"],
                "resets": self.vio_state["reset"], "corr_std": round(self.fusion.correction_std, 3),
                "fixes": self.fusion.accepted},
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
                     seed: int | None = None, nav_source: str | None = None,
                     floor_map: str | None = None, teach_out: str | None = None) -> MissionRunner:
    cfg = load_sim_config(config_path, overrides)
    if nav_source:
        cfg["nav_source"] = nav_source
    if floor_map:
        cfg["map_matching"]["map"] = floor_map
    if teach_out:
        cfg["map_matching"]["teach_out"] = teach_out
    if seed is not None:
        cfg["seed"] = seed
    mission = Mission.load(mission_name)
    world = World.load(mission.world)
    vehicle = SimVehicle(cfg, world)
    if log_path == "auto":
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / f"{mission.name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    return MissionRunner(cfg, mission, world, vehicle, log_path, landmark_correction, annotate, fault_schedule)
