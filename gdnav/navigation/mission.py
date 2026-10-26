"""Mission files and the waypoint_manager that tracks progress through them."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gdnav.common.config import MISSIONS_DIR, load_yaml, resolve


@dataclass
class Waypoint:
    id: int
    x: float
    y: float
    landmark: str | None = None
    hold_s: float = 2.0
    alt: float | None = None

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Mission:
    name: str
    world: str
    altitude_m: float
    waypoints: list[Waypoint]
    on_finish: str = "return_to_start"          # return_to_start | land | hold
    on_lost_landmark: str = "hold_then_continue"  # hold_then_continue | return_to_start | land
    precision_land_landmark: str | None = None
    frame: str = "local_ned"
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Mission":
        d = load_yaml(resolve(name_or_path, MISSIONS_DIR))
        if d.get("frame", "local_ned") != "local_ned":
            raise ValueError("Only local_ned missions are supported (no GPS).")
        wps = [Waypoint(id=w.get("id", i + 1), x=float(w["x"]), y=float(w["y"]),
                        landmark=w.get("landmark"), hold_s=float(w.get("hold_s", 2.0)),
                        alt=w.get("alt"))
               for i, w in enumerate(d["waypoints"])]
        return cls(name=d.get("name", Path(str(name_or_path)).stem), world=d.get("world", "indoor_arena"),
                   altitude_m=float(d.get("altitude_m", 2.0)), waypoints=wps,
                   on_finish=d.get("on_finish", "return_to_start"),
                   on_lost_landmark=d.get("on_lost_landmark", "hold_then_continue"),
                   precision_land_landmark=d.get("precision_land_landmark"), raw=d)

    def to_dict(self) -> dict:
        return {"name": self.name, "altitude_m": self.altitude_m, "on_finish": self.on_finish,
                "waypoints": [w.__dict__ for w in self.waypoints],
                "precision_land_landmark": self.precision_land_landmark}


class WaypointManager:
    def __init__(self, mission: Mission):
        self.mission = mission
        self.index = 0
        self.results: list[dict] = []

    @property
    def current(self) -> Waypoint | None:
        if self.index < len(self.mission.waypoints):
            return self.mission.waypoints[self.index]
        return None

    @property
    def done(self) -> bool:
        return self.index >= len(self.mission.waypoints)

    def altitude(self, wp: Waypoint | None = None) -> float:
        if wp is not None and wp.alt is not None:
            return float(wp.alt)
        return self.mission.altitude_m

    def advance(self, result: dict) -> None:
        self.results.append(result)
        self.index += 1

    def skip_remaining(self) -> None:
        self.index = len(self.mission.waypoints)
