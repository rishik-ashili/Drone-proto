"""Common flight log format shared by every phase (JSON Lines).

Each line is one JSON object with a ``kind`` field:
  * ``meta``   - first line: config, world and mission used for the run
  * ``tick``   - navigation-rate sample: timestamps, setpoints, true / estimated
                 position, detections and health
  * ``event``  - state transitions, failsafes, fault injections, waypoint arrivals
  * ``summary``- final metrics, written when the run ends

The same file feeds ``gdnav.tools.report`` (plots and metrics) and
``gdnav.tools.replay`` (plays a flight back into the dashboard).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, np.ndarray):
        return [round(float(x), 4) for x in o.ravel()]
    if isinstance(o, (np.floating,)):
        return round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "value"):
        return o.value
    raise TypeError(f"not serialisable: {type(o)}")


class FlightLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._f = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "w", encoding="utf-8")

    def write(self, kind: str, **data: Any) -> None:
        if not self._f:
            return
        data["kind"] = kind
        data.setdefault("wall", round(time.time(), 3))
        self._f.write(json.dumps(data, default=_default) + "\n")

    def flush(self) -> None:
        if self._f:
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def read_log(path: str | Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"meta": [], "tick": [], "event": [], "summary": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.setdefault(rec.get("kind", "other"), []).append(rec)
    return out
