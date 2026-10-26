"""Classical CV fallback (no ML): HSV colour segmentation for the coloured pads.

Used when no YOLO model is available, and as the "CPU-only fallback" of the
plan for very small companion boards. It cannot recognise the helipad.
"""
from __future__ import annotations

import cv2
import numpy as np

from gdnav.perception.detections import Detection

# (class, list of HSV ranges, min fill ratio of the bounding box)
_RULES = [
    ("pad_red", [((0, 170, 90), (8, 255, 255)), ((170, 170, 90), (180, 255, 255))], 0.6),
    ("pad_blue", [((100, 150, 80), (125, 255, 255))], 0.75),
    ("cross_yellow", [((18, 120, 120), (35, 255, 255))], 0.3),
]


class ColorDetector:
    def __init__(self, min_area_px: int = 600):
        self.min_area = min_area_px
        self.last_ms = 0.0

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for cls, ranges, min_fill in _RULES:
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # outer contours only, filled: a pad with a white ring inside is still one shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                if not (0.6 < rw / max(rh, 1) < 1.6):
                    continue
                fill = area / max(rw * rh, 1.0)  # solidity vs. the rotated bounding box
                if fill < min_fill:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                conf = float(min(0.95, 0.5 + fill * 0.45))
                out.append(Detection(cls, conf, (float(x), float(y), float(x + w), float(y + h)), "color"))
        return out
