"""mission_state_machine: TAKEOFF -> GOTO_WAYPOINT -> SEARCH_LANDMARK -> ALIGN
-> NEXT_WAYPOINT -> ... -> RETURN_TO_START -> PRECISION_LAND -> LAND.

The state machine runs at the navigation rate on the companion computer and
streams Offboard setpoints to the flight controller. It only ever uses the
*estimated* state and confirmed landmark observations - never ground truth.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Callable

import numpy as np

from gdnav.interface.vehicle import Telemetry, VehicleInterface
from gdnav.navigation.geofence import Geofence
from gdnav.navigation.local_planner import LocalPlanner
from gdnav.navigation.mission import Mission, WaypointManager
from gdnav.perception.detections import LandmarkObservation
from gdnav.safety.health_monitor import HealthMonitor
from gdnav.sim.flight_controller import Setpoint


class State(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    TAKEOFF = "TAKEOFF"
    GOTO_WAYPOINT = "GOTO_WAYPOINT"
    SEARCH_LANDMARK = "SEARCH_LANDMARK"
    ALIGN = "ALIGN"
    NEXT_WAYPOINT = "NEXT_WAYPOINT"
    RETURN_TO_START = "RETURN_TO_START"
    PRECISION_LAND = "PRECISION_LAND"
    FAILSAFE_HOLD = "FAILSAFE_HOLD"
    LAND = "LAND"
    LANDED = "LANDED"
    ABORTED = "ABORTED"


FLYING_STATES = {State.TAKEOFF, State.GOTO_WAYPOINT, State.SEARCH_LANDMARK, State.ALIGN,
                 State.NEXT_WAYPOINT, State.RETURN_TO_START, State.PRECISION_LAND, State.FAILSAFE_HOLD}
VISION_STATES = {State.SEARCH_LANDMARK, State.ALIGN, State.PRECISION_LAND}


def _spiral(step: float, rings: int = 3) -> list[np.ndarray]:
    """Expanding square search pattern around the waypoint."""
    pts = [np.zeros(2)]
    x = y = 0
    dx, dy = 1, 0
    seg = 1
    while len(pts) < (2 * rings + 1) ** 2:
        for _ in range(2):
            for _ in range(seg):
                x, y = x + dx, y + dy
                pts.append(np.array([x * step, y * step]))
            dx, dy = -dy, dx
        seg += 1
    return pts


class MissionStateMachine:
    def __init__(self, nav_cfg: dict, safety_cfg: dict, mission: Mission, vehicle: VehicleInterface,
                 planner: LocalPlanner, geofence: Geofence, health: HealthMonitor,
                 event_cb: Callable[[str, str, dict], None], truth_fn: Callable[[], dict | None] = lambda: None,
                 vision: bool = True):
        self.c = nav_cfg
        # vision=False: fly on the GPS-free estimate alone (no landmark alignment / precision
        # landing) - the "without our AI" baseline
        self.vision = vision
        self.sc = safety_cfg
        self.mission = mission
        self.mgr = WaypointManager(mission)
        self.v = vehicle
        self.planner = planner
        self.fence = geofence
        self.health = health
        self.emit = event_cb
        self.truth_fn = truth_fn

        self.state = State.IDLE
        self.state_t = 0.0
        self.t = 0.0
        self.home = np.zeros(2)
        self.home_yaw = 0.0
        self.yaw_sp = 0.0
        self.alt = mission.altitude_m
        self.visible: dict[str, tuple[float, LandmarkObservation]] = {}
        self.last_sp: Setpoint | None = None
        self.resume_state: State | None = None
        self.hold_pos = np.zeros(3)
        self.hold_reason = ""
        self._search_pts: list[np.ndarray] = []
        self._search_i = 0
        self._search_leg_t = 0.0
        self._aligned_since: float | None = None
        self._align_int = np.zeros(2)
        self._last_seen_t = 0.0
        self._align_retries = 0
        self._land_sent = False
        self._arrived_t: float | None = None
        self._clear_since: float | None = None
        self._last_lm_xy: np.ndarray | None = None
        self._dwell_done = False
        self._found = False
        self.rts_result: dict | None = None
        self.mission_complete = False

    # ------------------------------------------------------------ helpers
    def start(self) -> None:
        if self.state in (State.IDLE, State.LANDED, State.ABORTED):
            if self.state != State.IDLE:  # flying the mission again from where it landed
                self.mgr = WaypointManager(self.mission)
                self.mission_complete = False
                self.rts_result = None
            self._enter(State.PREFLIGHT, "mission start requested")

    def _enter(self, s: State, reason: str = "") -> None:
        if s == self.state:
            return
        prev = self.state
        self.state = s
        self.state_t = self.t
        self.emit("state", f"{prev.value} -> {s.value}" + (f" ({reason})" if reason else ""),
                  {"from": prev.value, "to": s.value, "reason": reason})

    def _since(self) -> float:
        return self.t - self.state_t

    def _send(self, sp: Setpoint) -> None:
        if sp.kind == "position":
            sp.pos = self.fence.clamp(sp.pos)
        self.last_sp = sp
        self.v.send_setpoint(sp)

    def _pos_sp(self, xy, alt, yaw=None) -> None:
        self._send(Setpoint("position", pos=np.array([xy[0], xy[1], -alt]),
                            yaw=self.yaw_sp if yaw is None else yaw))

    def _vel_sp(self, vxy, telem: Telemetry, alt, yaw=None) -> None:
        vz = float(np.clip(1.0 * (-alt - telem.pos[2]), -0.8, 0.6))
        self._send(Setpoint("velocity", vel=np.array([vxy[0], vxy[1], vz]),
                            yaw=self.yaw_sp if yaw is None else yaw))

    def observe(self, t: float, observations: list[LandmarkObservation]) -> None:
        for o in observations:
            if o.confirmed:
                self.visible[o.name] = (t, o)

    def _obs(self, name: str | None, max_age: float = 0.6) -> LandmarkObservation | None:
        if not self.vision or not name or name not in self.visible:
            return None
        t, o = self.visible[name]
        return o if self.t - t <= max_age else None

    def _record(self, telem: Telemetry, found: bool) -> None:
        wp = self.mgr.current
        truth = self.truth_fn()
        res = {"id": wp.id, "landmark": wp.landmark, "found": found, "vision": self.vision, "t": round(self.t, 2),
               "est_error_m": round(float(np.linalg.norm(telem.pos[:2] - wp.xy)), 3)}
        if truth is not None:
            res["reach_error_m"] = round(float(np.linalg.norm(truth["pos"][:2] - wp.xy)), 3)
        self.mgr.advance(res)
        msg = f"Waypoint {wp.id} done" + (f" - true reach error {res['reach_error_m']:.2f} m" if truth else "")
        self.emit("waypoint", msg, res)

    # --------------------------------------------------------------- step
    def step(self, t: float, telem: Telemetry, depth=None) -> None:
        self.t = t
        self.telem = telem
        self.depth = depth
        flying = self.state in FLYING_STATES
        action = self.health.worst_action()

        # vehicle-initiated landing (e.g. PX4 critical battery failsafe)
        if flying and telem.mode == "LAND" and self.state not in (State.LAND,) and telem.received_t >= t - 0.5:
            self._enter(State.LAND, "flight controller is landing")
        elif flying:
            if action == "land" and self.state != State.LAND:
                self._enter(State.LAND, self._fault_text("land"))
            elif action == "rts" and self.state not in (State.RETURN_TO_START, State.PRECISION_LAND,
                                                        State.LAND, State.FAILSAFE_HOLD):
                self.mgr.skip_remaining()
                self._enter(State.RETURN_TO_START, self._fault_text("rts"))
            elif action == "hold" and self.state != State.FAILSAFE_HOLD:
                self._clear_since = None
                self.resume_state = self.state
                self.hold_pos = telem.pos.copy()
                self.hold_reason = self._fault_text("hold")
                self._enter(State.FAILSAFE_HOLD, self.hold_reason)

        getattr(self, "_s_" + self.state.value.lower())(telem)

    def _fault_text(self, action: str) -> str:
        return ", ".join(f.message for f in self.health.faults.values() if f.action == action)

    # ------------------------------------------------------------- states
    def _s_idle(self, telem):
        pass

    def _s_aborted(self, telem):
        pass

    def _s_landed(self, telem):
        pass

    def _s_preflight(self, telem: Telemetry):
        if self._since() < 0.05:
            self.home = telem.pos[:2].copy()
            self.home_yaw = float(telem.att[2])
            self.yaw_sp = self.home_yaw
        if telem.battery < 0.5:
            self._enter(State.ABORTED, f"pre-flight: battery {telem.battery*100:.0f} %")
            return
        if telem.pos_std > 0.5:
            self._enter(State.ABORTED, "pre-flight: position estimate not ready")
            return
        # PX4 only accepts Offboard once setpoints are streaming
        self._pos_sp(self.home, self.alt)
        if self._since() > 0.6:
            if self.v.arm() and self.v.start_offboard():
                self._enter(State.TAKEOFF, "armed, offboard engaged")
            elif self._since() > 5.0:
                self._enter(State.ABORTED, "could not arm / engage offboard")

    def _s_takeoff(self, telem: Telemetry):
        self._pos_sp(self.home, self.alt)
        if -telem.pos[2] > self.alt - 0.15:
            if self.mgr.done:
                self._finish()
            else:
                self._enter(State.GOTO_WAYPOINT, f"to waypoint {self.mgr.current.id}")
        elif self._since() > self.c["takeoff_timeout_s"]:
            self._enter(State.LAND, "takeoff timeout")

    def _fly_to(self, telem: Telemetry, target: np.ndarray) -> float:
        d_vec = target - telem.pos[:2]
        d = float(np.linalg.norm(d_vec))
        if d > 1.0:
            if self.c.get("face_travel_direction", True):
                self.yaw_sp = math.atan2(d_vec[1], d_vec[0])
            v_des = d_vec / d * min(self.c["cruise_speed"], 0.8 * d)
            v = self.planner.adjust(v_des, telem.pos[:2], float(telem.att[2]), self.depth)
            self._vel_sp(v, telem, self.alt)
        else:
            self.planner.active = False
            self._pos_sp(target, self.alt)
        return d

    def _s_goto_waypoint(self, telem: Telemetry):
        wp = self.mgr.current
        if wp is None:
            self._finish()
            return
        d = self._fly_to(telem, wp.xy)
        if wp.landmark and self._obs(wp.landmark) is not None and d < 2.5:
            self._start_align()
            return
        speed = float(np.linalg.norm(telem.vel[:2]))
        if d < self.c["arrival_tolerance_m"] and speed < 0.4:
            if wp.landmark and self.vision:
                self._start_search("arrived, landmark not yet confirmed")
            else:
                self._dwell_done = False
                self._enter(State.NEXT_WAYPOINT, "arrived")

    def _start_search(self, reason: str):
        self._search_pts = _spiral(self.c["search_step_m"], 2)
        self._search_i = 0
        self._search_leg_t = self.t
        self._enter(State.SEARCH_LANDMARK, reason)

    def _s_search_landmark(self, telem: Telemetry):
        wp = self.mgr.current
        if self._obs(wp.landmark) is not None:
            self._start_align()
            return
        if self._since() > self.c["search_timeout_s"]:
            self.emit("warn", f"Landmark {wp.landmark} not found - {self.mission.on_lost_landmark}", {})
            if self.mission.on_lost_landmark == "return_to_start":
                self.mgr.skip_remaining()
                self._enter(State.RETURN_TO_START, "landmark lost")
            elif self.mission.on_lost_landmark == "land":
                self._enter(State.LAND, "landmark lost")
            else:
                self._found = False
                self._dwell_done = False
                self._enter(State.NEXT_WAYPOINT, "landmark not found, continuing")
            return
        target = wp.xy + self._search_pts[self._search_i % len(self._search_pts)]
        self._pos_sp(target, self.alt)
        if np.linalg.norm(target - telem.pos[:2]) < 0.3 or self.t - self._search_leg_t > 4.0:
            self._search_i += 1
            self._search_leg_t = self.t

    def _start_align(self):
        self._last_lm_xy = None
        self._aligned_since = None
        self._align_int[:] = 0
        self._last_seen_t = self.t
        self._enter(State.ALIGN, f"{self.mgr.current.landmark} confirmed")

    def _servo(self, o: LandmarkObservation, max_speed: float) -> np.ndarray:
        err = o.offset_ned
        self._align_int = np.clip(self._align_int + err * 0.05, -1.0, 1.0)
        v = self.c["align_kp"] * err + self.c["align_ki"] * self._align_int
        n = np.linalg.norm(v)
        return v * (max_speed / n) if n > max_speed else v

    def _s_align(self, telem: Telemetry):
        wp = self.mgr.current
        o = self._obs(wp.landmark)
        if o is None:
            # fly to where the landmark was last seen (it may have slipped out of view)
            self._pos_sp(self._last_lm_xy if self._last_lm_xy is not None else telem.pos[:2], self.alt)
            if self.t - self._last_seen_t > self.sc["detection_timeout_s"]:
                self._align_retries += 1
                if self._align_retries > 2:
                    self._found = False
                    self._dwell_done = False
                    self._enter(State.NEXT_WAYPOINT, "detection lost repeatedly")
                else:
                    self._start_search("detection timeout")
            return
        self._last_seen_t = self.t
        self._last_lm_xy = telem.pos[:2] + o.offset_ned
        self._vel_sp(self._servo(o, self.c["align_max_speed"]), telem, self.alt)
        if np.linalg.norm(o.offset_ned) < self.c["align_tolerance_m"]:
            if self._aligned_since is None:
                self._aligned_since = self.t
            elif self.t - self._aligned_since >= wp.hold_s:
                self._found = True
                self._dwell_done = True
                self._enter(State.NEXT_WAYPOINT, f"aligned over {wp.landmark} for {wp.hold_s:.0f} s")
        elif np.linalg.norm(o.offset_ned) > 2 * self.c["align_tolerance_m"]:
            self._aligned_since = None

    def _s_next_waypoint(self, telem: Telemetry):
        wp = self.mgr.current
        self._pos_sp(telem.pos[:2] if self._dwell_done else wp.xy, self.alt)
        dwell = 0.3 if self._dwell_done else wp.hold_s
        if self._since() >= dwell:
            self._record(telem, self._found if wp.landmark else True)
            self._found = False
            self._align_retries = 0
            if self.mgr.done:
                self._finish()
            else:
                self._enter(State.GOTO_WAYPOINT, f"to waypoint {self.mgr.current.id}")

    def _finish(self):
        self.mission_complete = True
        how = self.mission.on_finish
        if how == "return_to_start":
            self._enter(State.RETURN_TO_START, "all waypoints done")
        elif how == "land":
            self._enter(State.LAND, "all waypoints done")
        else:
            self.resume_state = None
            self.hold_pos = self.telem.pos.copy()
            self.hold_reason = "mission finished (hold)"
            self._enter(State.FAILSAFE_HOLD, "mission finished")

    def _s_return_to_start(self, telem: Telemetry):
        d = self._fly_to(telem, self.home)
        lm = self.mission.precision_land_landmark if self.vision else None
        if lm and self._obs(lm) is not None and d < 2.0:
            self._enter(State.PRECISION_LAND, f"home pad '{lm}' confirmed")
            return
        if d < self.c["arrival_tolerance_m"] and np.linalg.norm(telem.vel[:2]) < 0.4:
            if self._arrived_t is None or self._arrived_t < self.state_t:
                self._arrived_t = self.t
            # with a home pad, hover a few seconds so it can be confirmed before landing blind
            if not lm or self.t - self._arrived_t > 8.0:
                self._enter(State.LAND, "home reached" + (", home pad not seen" if lm else ""))

    def _s_precision_land(self, telem: Telemetry):
        o = self._obs(self.mission.precision_land_landmark)
        alt = -telem.pos[2]
        if o is not None:
            self._last_seen_t = self.t
            v = self._servo(o, 0.5)
            err = float(np.linalg.norm(o.offset_ned))
            vz = 0.35 if err < 0.25 else 0.0
            self._send(Setpoint("velocity", vel=np.array([v[0], v[1], vz]), yaw=self.yaw_sp))
        else:
            self._send(Setpoint("velocity", vel=np.array([0.0, 0.0, 0.2]), yaw=self.yaw_sp))
        if alt < 0.6 or self.t - self._last_seen_t > 4.0:
            self._enter(State.LAND, "final descent")

    def _s_failsafe_hold(self, telem: Telemetry):
        self._pos_sp(self.hold_pos[:2], -self.hold_pos[2] if self.hold_pos[2] < -0.3 else self.alt)
        action = self.health.worst_action()
        if self.resume_state is None:  # finished-mission hold: nothing to resume
            return
        if action not in ("none", "warn"):
            self._clear_since = None
        elif self._clear_since is None:
            self._clear_since = self.t
        if self._clear_since is not None and self.t - self._clear_since >= 1.0:  # hysteresis
            if telem.mode != "OFFBOARD":
                if not self.v.start_offboard():
                    return
            self.emit("info", f"Fault cleared - resuming {self.resume_state.value}", {})
            resume = self.resume_state
            self.resume_state = None
            self._enter(resume, "fault cleared")
            return
        since = self.health.hold_faults_since()
        if since is not None and self.t - since > self.sc["hold_then_land_s"]:
            self._enter(State.LAND, f"fault persisted {self.sc['hold_then_land_s']:.0f} s")

    def _s_land(self, telem: Telemetry):
        if not self._land_sent or (telem.mode != "LAND" and telem.armed and self._since() > 1.0):
            self.v.land()
            self._land_sent = True
        if not telem.armed and telem.landed:
            self._land_sent = False
            truth = self.truth_fn()
            res = {"est_offset_m": round(float(np.linalg.norm(telem.pos[:2] - self.home)), 3)}
            if truth is not None:
                err = float(np.linalg.norm(truth["pos"][:2] - self.home))
                res["rts_error_m"] = round(err, 3)
                res["distance_m"] = round(truth["distance"], 2)
                res["rts_error_pct"] = round(100 * err / max(truth["distance"], 1e-3), 3)
            self.rts_result = res
            self.emit("landed", "Landed" + (f" - {res['rts_error_m']:.2f} m from takeoff point"
                                            if "rts_error_m" in res else ""), res)
            self._enter(State.LANDED, "disarmed")
