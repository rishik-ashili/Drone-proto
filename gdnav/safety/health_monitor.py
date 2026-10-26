"""health_monitor: watches battery, link, camera, detections and tracking
quality, and recommends a failsafe action.

Failsafe chain (from the plan): fault -> HOLD -> LAND if it persists.
Battery has its own ladder: return-to-start, then land.
"""
from __future__ import annotations

from dataclasses import dataclass

from gdnav.interface.vehicle import Telemetry

ACTION_RANK = {"none": 0, "warn": 1, "hold": 2, "rts": 3, "land": 4}


@dataclass
class Fault:
    name: str
    action: str
    message: str
    since: float


class HealthMonitor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.faults: dict[str, Fault] = {}
        self.last_frame_t = 0.0
        self.image_stats: tuple | None = None  # (mean, std) of the latest processed frame

    def _set(self, active: bool, name: str, action: str, msg: str, now: float) -> None:
        if active:
            f = self.faults.get(name)
            if f is None or f.action != action:
                self.faults[name] = Fault(name, action, msg, f.since if f else now)
            else:
                f.message = msg
        else:
            self.faults.pop(name, None)

    def evaluate(self, now: float, telem: Telemetry, frame_t: float | None, image_stats: tuple | None,
                 flying: bool, vision_needed: bool, pos_in_fence: bool) -> dict[str, Fault]:
        c = self.cfg
        # link: telemetry stops arriving
        link_age = now - telem.received_t
        self._set(link_age > c["link_timeout_s"], "link", "hold",
                  f"No telemetry for {link_age:.1f} s", now)

        # battery ladder
        pct = telem.battery * 100
        if pct < c["battery_land_pct"]:
            self._set(True, "battery", "land", f"Battery {pct:.0f} % - landing", now)
        elif pct < c["battery_rts_pct"]:
            self._set(True, "battery", "rts", f"Battery {pct:.0f} % - returning", now)
        else:
            self._set(False, "battery", "", "", now)

        # camera: frames stop, or image is black / uniform (lens covered)
        if frame_t is not None:
            self.last_frame_t = frame_t
        if image_stats is not None:
            self.image_stats = image_stats
        cam_age = now - self.last_frame_t
        st = self.image_stats
        covered = st is not None and (st[0] < 18 or st[1] < 4)
        cam_bad = cam_age > c["camera_timeout_s"] or covered
        msg = "Camera covered / dark" if covered else f"No camera frame for {cam_age:.1f} s"
        self._set(cam_bad and flying, "camera", "hold" if vision_needed else "warn", msg, now)

        # position estimate quality (EKF eph)
        self._set(flying and telem.pos_std > c["max_pos_std_m"], "position_quality", "hold",
                  f"Position uncertainty {telem.pos_std:.2f} m", now)

        self._set(flying and not pos_in_fence, "geofence", "rts", "Outside software geofence", now)

        if telem.failsafe and flying:
            self._set(True, "vehicle_failsafe", "warn", f"PX4 failsafe: {telem.failsafe}", now)
        else:
            self._set(False, "vehicle_failsafe", "", "", now)
        return self.faults

    def worst_action(self) -> str:
        best = "none"
        for f in self.faults.values():
            if ACTION_RANK[f.action] > ACTION_RANK[best]:
                best = f.action
        return best

    def hold_faults_since(self) -> float | None:
        ts = [f.since for f in self.faults.values() if f.action == "hold"]
        return min(ts) if ts else None
