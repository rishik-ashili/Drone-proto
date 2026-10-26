"""Software geofence kept in the navigation code in addition to PX4's own."""
from __future__ import annotations

import numpy as np


class Geofence:
    def __init__(self, x_min, x_max, y_min, y_max, max_alt, margin=0.5):
        self.b = (x_min + margin, x_max - margin, y_min + margin, y_max - margin)
        self.max_alt = max_alt

    def clamp(self, xyz: np.ndarray) -> np.ndarray:
        out = xyz.copy()
        out[0] = np.clip(out[0], self.b[0], self.b[1])
        out[1] = np.clip(out[1], self.b[2], self.b[3])
        out[2] = max(out[2], -self.max_alt)
        return out

    def breached(self, pos: np.ndarray, slack: float = 0.3) -> bool:
        x, y = pos[0], pos[1]
        return (x < self.b[0] - slack or x > self.b[1] + slack or y < self.b[2] - slack
                or y > self.b[3] + slack or -pos[2] > self.max_alt + 0.5)
