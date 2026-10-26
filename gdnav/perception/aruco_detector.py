"""AprilTag / ArUco anchors (OpenCV): no training needed, very reliable."""
from __future__ import annotations

import cv2
import numpy as np

from gdnav.perception.detections import Detection


class ArucoDetector:
    def __init__(self, dictionary: int = cv2.aruco.DICT_4X4_50):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out: list[Detection] = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            pts = c.reshape(4, 2)
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            out.append(Detection("aruco_tag", 1.0, (float(x1), float(y1), float(x2), float(y2)),
                                 "aruco", tag_id=int(i), corners=pts))
        return out
