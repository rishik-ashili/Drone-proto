"""Compose demo video frames (onboard camera + top-down map + HUD) with OpenCV.

Used by ``scripts/run_sim.py --record demo.mp4`` to produce a shareable video
of a mission without needing a browser or screen recorder.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess

import cv2
import numpy as np

GREEN, CYAN, ORANGE, WHITE, GREY = (132, 220, 61), (255, 182, 57), (67, 159, 255), (243, 237, 230), (171, 154, 138)


class VideoComposer:
    def __init__(self, runner, path: str, fps: int = 20, size: tuple[int, int] = (1280, 720)):
        self.r = runner
        self.W, self.H = size
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        w = runner.world
        self.map_px = self.H - 80
        s = self.map_px / max(w.x_max - w.x_min, w.y_max - w.y_min)
        self.s = s
        self.base = cv2.resize(w.texture, (int((w.y_max - w.y_min) * s), int((w.x_max - w.x_min) * s)),
                               interpolation=cv2.INTER_AREA)
        self.base = (self.base * 0.75).astype(np.uint8)

    def P(self, n, e) -> tuple[int, int]:
        w = self.r.world
        return int((e - w.y_min) * self.s), int((w.x_max - n) * self.s)

    def _map(self) -> np.ndarray:
        img = self.base.copy()
        m = self.r.mission
        pts = [self.P(0, 0)] + [self.P(wp.x, wp.y) for wp in m.waypoints]
        if m.on_finish == "return_to_start":
            pts.append(self.P(0, 0))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (200, 200, 200), 1, cv2.LINE_AA)
        tr = self.r.trail
        for idx, col in ((7, ORANGE), (4, CYAN), (1, GREEN)):
            if len(tr) > 1:
                poly = np.array([self.P(p[idx], p[idx + 1]) for p in tr], np.int32)
                cv2.polylines(img, [poly], False, col, 2, cv2.LINE_AA)
        for i, wp in enumerate(m.waypoints):
            c = self.P(wp.x, wp.y)
            done = i < self.r.sm.mgr.index
            cv2.circle(img, c, 10, GREEN if done else (40, 30, 20), -1, cv2.LINE_AA)
            cv2.circle(img, c, 10, WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, str(wp.id), (c[0] - 4, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        if self.r.is_sim:
            t = self.r.vehicle.truth()
            c = self.P(t["pos"][0], t["pos"][1])
            yaw = t["att"][2]
            tri = np.array([[0, -12], [8, 9], [0, 4], [-8, 9]], float)
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            tri = (tri @ R.T + c).astype(np.int32)
            cv2.fillPoly(img, [tri], GREEN, cv2.LINE_AA)
            cv2.circle(img, self.P(*t["dr_pos"][:2]), 5, ORANGE, -1, cv2.LINE_AA)
        est = self.r.vehicle.telemetry().pos
        cv2.circle(img, self.P(est[0], est[1]), 5, CYAN, -1, cv2.LINE_AA)
        return img

    def write(self) -> None:
        r = self.r
        frame = np.full((self.H, self.W, 3), (19, 14, 10), np.uint8)
        mp = self._map()
        mh, mw = mp.shape[:2]
        frame[60:60 + mh, 20:20 + mw] = mp
        cam_w = self.W - mw - 60
        cam_h = int(cam_w * 3 / 4)
        if r.last_perception is not None and r.last_perception.annotated is not None:
            cam = cv2.resize(r.last_perception.annotated, (cam_w, cam_h))
            frame[60:60 + cam_h, 40 + mw:40 + mw + cam_w] = cam
        m = r.metrics()
        put = lambda t, x, y, c=WHITE, s=0.6: cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
        put("GPS-denied navigation | digital twin", 20, 36, WHITE, 0.8)
        put(f"t={r.t:6.1f}s  {r.sm.state.value}", self.W - 420, 36, CYAN, 0.75)
        x0, y0 = 40 + mw, 60 + cam_h + 40
        put(f"Position error, vision-corrected: {m['est_error_now_m']:.2f} m", x0, y0, CYAN)
        put(f"Position error, dead reckoning:   {m['dr_error_now_m']:.2f} m", x0, y0 + 30, ORANGE)
        reach = m["reach_error_mean_m"]
        put(f"Waypoints {m['waypoints_done']}/{m['waypoints_total']}" +
            (f"  mean reach error {reach:.2f} m" if reach is not None else ""), x0, y0 + 60, GREEN)
        faults = ", ".join(f.name for f in r.health.faults.values())
        if faults:
            put(f"FAULTS: {faults}", x0, y0 + 90, (93, 93, 255))
        put("true", 30, self.H - 14, GREEN, 0.5)
        put("estimate (vision)", 80, self.H - 14, CYAN, 0.5)
        put("dead reckoning", 230, self.H - 14, ORANGE, 0.5)
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
        # OpenCV writes MPEG-4 Part 2; re-encode to H.264 (plays everywhere) if ffmpeg is installed
        if shutil.which("ffmpeg"):
            tmp = self.path + ".h264.mp4"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.path, "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", tmp])
            if r.returncode == 0:
                os.replace(tmp, self.path)
