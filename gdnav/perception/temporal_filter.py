"""Temporal filtering: a landmark is only 'confirmed' after it has been seen on
several consecutive frames at a consistent position. This stops single-frame
false positives from ever reaching the navigation logic."""
from __future__ import annotations

import numpy as np


class TemporalFilter:
    def __init__(self, frames_required: int = 3, max_jump_m: float = 0.6, max_misses: int = 1):
        self.n = frames_required
        self.max_jump = max_jump_m
        self.max_misses = max_misses
        self.tracks: dict[str, dict] = {}

    def update(self, observations: dict[str, np.ndarray]) -> set[str]:
        """observations: landmark name -> implied landmark world xy this frame.
        Returns the names that are confirmed on this frame."""
        confirmed = set()
        for name, xy in observations.items():
            tr = self.tracks.get(name)
            if tr is None or np.linalg.norm(xy - tr["xy"]) > self.max_jump:
                tr = {"count": 0, "misses": 0, "xy": xy}
            tr["count"] += 1
            tr["misses"] = 0
            tr["xy"] = 0.5 * tr["xy"] + 0.5 * xy
            self.tracks[name] = tr
            if tr["count"] >= self.n:
                confirmed.add(name)
        for name in list(self.tracks):
            if name not in observations:
                self.tracks[name]["misses"] += 1
                if self.tracks[name]["misses"] > self.max_misses:
                    del self.tracks[name]
        return confirmed

    def reset(self) -> None:
        self.tracks.clear()
