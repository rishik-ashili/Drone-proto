from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    source: str                              # yolo | aruco | color | injected
    tag_id: int | None = None
    corners: np.ndarray | None = None        # 4x2 (ArUco)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class LandmarkObservation:
    """A detection associated with a mapped landmark and turned into metres."""
    name: str
    cls: str
    source: str
    conf: float
    offset_ned: np.ndarray        # landmark position relative to the drone (m)
    fix_xy: np.ndarray            # implied drone position = landmark - offset
    std: float                    # 1-sigma of the fix (m)
    yaw_fix: float | None = None  # implied drone heading (ArUco only)
    confirmed: bool = False
    accepted: bool | None = None  # set once the estimator gates it
    extra: dict = field(default_factory=dict)
