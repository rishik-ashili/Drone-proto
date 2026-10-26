"""Arena model: procedural floor texture, landmarks, distractors and obstacles.

The floor is a single large RGB texture (1 cm per pixel by default). The
simulated downward camera is rendered from it with a homography, so every
detector in the pipeline (ArUco, YOLO, colour fallback) runs on real pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gdnav.common.config import WORLDS_DIR, load_yaml, resolve

# Classes the custom YOLO landmark model is trained on. ArUco tags are also a
# YOLO class so the network can find tags that are too blurred to decode.
YOLO_CLASSES = ["helipad", "pad_red", "pad_blue", "cross_yellow", "aruco_tag"]
TYPE_TO_CLASS = {
    "helipad": "helipad",
    "pad_red": "pad_red",
    "pad_blue": "pad_blue",
    "cross_yellow": "cross_yellow",
    "aruco": "aruco_tag",
}
ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Landmark:
    name: str
    type: str
    x: float
    y: float
    size: float
    yaw_deg: float = 0.0
    id: int | None = None

    @property
    def cls(self) -> str:
        return TYPE_TO_CLASS[self.type]

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Obstacle:
    name: str
    x: float
    y: float
    radius: float
    height: float


@dataclass
class World:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res: float
    landmarks: list[Landmark]
    obstacles: list[Obstacle]
    surface: str = "concrete"
    grid_m: float = 2.0
    seed: int = 0
    distractors: int = 0
    texture: np.ndarray | None = field(default=None, repr=False)

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(cls, name_or_path: str | Path, build_texture: bool = True) -> "World":
        d = load_yaml(resolve(name_or_path, WORLDS_DIR))
        b = d["bounds"]
        w = cls(
            name=d.get("name", "world"),
            x_min=b["x_min"], x_max=b["x_max"], y_min=b["y_min"], y_max=b["y_max"],
            res=d.get("texture_res_m", 0.01),
            landmarks=[Landmark(**lm) for lm in d.get("landmarks", [])],
            obstacles=[Obstacle(**o) for o in d.get("obstacles", [])],
            surface=d.get("surface", "concrete"),
            grid_m=d.get("grid_m", 2.0),
            seed=d.get("seed", 0),
            distractors=d.get("distractors", 0),
        )
        if build_texture:
            w.texture = w.build_texture()
        return w

    def landmark(self, name: str) -> Landmark | None:
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bounds": [self.x_min, self.x_max, self.y_min, self.y_max],
            "landmarks": [lm.__dict__ for lm in self.landmarks],
            "obstacles": [o.__dict__ for o in self.obstacles],
        }

    # --------------------------------------------------------------- geometry
    @property
    def tex_shape(self) -> tuple[int, int]:
        rows = int(round((self.x_max - self.x_min) / self.res))
        cols = int(round((self.y_max - self.y_min) / self.res))
        return rows, cols

    def world_to_tex(self, x: float, y: float) -> tuple[float, float]:
        """World (north, east) -> texture (col, row). North is up in the texture."""
        return (y - self.y_min) / self.res, (self.x_max - x) / self.res

    def tex_from_world_matrix(self) -> np.ndarray:
        """3x3 matrix T with [x, y, 1]^T = T @ [col, row, 1]^T."""
        r = self.res
        return np.array([[0.0, -r, self.x_max], [r, 0.0, self.y_min], [0.0, 0.0, 1.0]])

    def inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x_min + margin <= x <= self.x_max - margin
                and self.y_min + margin <= y <= self.y_max - margin)

    # ---------------------------------------------------------------- texture
    def build_texture(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        rows, cols = self.tex_shape
        tex = floor_texture(rows, cols, self.surface, rng)
        px_per_m = 1.0 / self.res

        # painted measurement grid
        if self.grid_m > 0:
            line = max(1, int(0.02 * px_per_m))
            colour = (200, 200, 190) if self.surface != "grass" else (225, 225, 225)
            gx = self.x_min + (-self.x_min % self.grid_m)
            while gx <= self.x_max:
                _, r = self.world_to_tex(gx, 0)
                cv2.line(tex, (0, int(r)), (cols - 1, int(r)), colour, line, cv2.LINE_AA)
                gx += self.grid_m
            gy = self.y_min + (-self.y_min % self.grid_m)
            while gy <= self.y_max:
                c, _ = self.world_to_tex(0, gy)
                cv2.line(tex, (int(c), 0), (int(c), rows - 1), colour, line, cv2.LINE_AA)
                gy += self.grid_m

        keepout = [(lm.x, lm.y, lm.size) for lm in self.landmarks]
        for _ in range(self.distractors):
            for _try in range(30):
                x = rng.uniform(self.x_min + 0.5, self.x_max - 0.5)
                y = rng.uniform(self.y_min + 0.5, self.y_max - 0.5)
                if all(math.hypot(x - kx, y - ky) > ks + 0.8 for kx, ky, ks in keepout):
                    break
            s = rng.uniform(0.3, 0.9)
            patch = random_distractor(int(s * px_per_m), rng)
            paste(tex, patch, *self.world_to_tex(x, y), rng.uniform(0, 360))
            keepout.append((x, y, s))

        for ob in self.obstacles:
            c, r = self.world_to_tex(ob.x, ob.y)
            rad = int(ob.radius * px_per_m)
            cv2.circle(tex, (int(c) + 6, int(r) + 6), rad, (40, 40, 40), -1, cv2.LINE_AA)  # shadow
            cv2.circle(tex, (int(c), int(r)), rad, (95, 90, 85), -1, cv2.LINE_AA)
            cv2.circle(tex, (int(c), int(r)), rad, (60, 60, 60), max(2, rad // 10), cv2.LINE_AA)

        for lm in self.landmarks:
            patch = render_landmark(lm.type, int(lm.size * px_per_m), lm.id, rng)
            paste(tex, patch, *self.world_to_tex(lm.x, lm.y), lm.yaw_deg)
        return tex


# ---------------------------------------------------------------------------
# Procedural drawing helpers (also used by the synthetic dataset generator)
# ---------------------------------------------------------------------------
def _noise(rows: int, cols: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    small = rng.standard_normal((max(2, rows // scale), max(2, cols // scale))).astype(np.float32)
    return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_CUBIC)


def floor_texture(rows: int, cols: int, surface: str, rng: np.random.Generator) -> np.ndarray:
    n = (0.5 * _noise(rows, cols, 400, rng) + 0.3 * _noise(rows, cols, 60, rng)
         + 0.2 * _noise(rows, cols, 8, rng))
    fine = rng.standard_normal((rows, cols)).astype(np.float32)
    if surface == "grass":
        base = np.array([60, 120, 70], np.float32)  # BGR
        amp, famp = 22.0, 14.0
    elif surface == "mat":
        base = np.array([80, 70, 60], np.float32)
        amp, famp = 14.0, 10.0
    else:  # concrete
        base = np.array([128, 132, 135], np.float32)
        amp, famp = 18.0, 9.0
    tex = base[None, None, :] + (amp * n + famp * fine)[:, :, None]
    if surface == "grass":
        tex[:, :, 1] += 10 * _noise(rows, cols, 20, rng)
    # sparse speckles / stains give optical flow something to track
    count = rows * cols // 2500
    ys = rng.integers(0, rows, count)
    xs = rng.integers(0, cols, count)
    vals = rng.uniform(-45, 45, count)
    tex[ys, xs, :] += vals[:, None]
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(tex, (3, 3), 0)


def render_landmark(kind: str, px: int, tag_id: int | None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a BGRA patch (alpha = 0 outside the landmark) of side ``px``."""
    px = max(px, 16)
    img = np.zeros((px, px, 4), np.uint8)
    c = px // 2
    if kind == "helipad":
        cv2.circle(img, (c, c), c - 1, (40, 40, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), int(c * 0.86), (235, 235, 235, 255), max(2, px // 28), cv2.LINE_AA)
        t = max(3, px // 9)
        h0, h1 = int(px * 0.30), int(px * 0.70)
        cv2.rectangle(img, (h0, h0), (h0 + t, h1), (235, 235, 235, 255), -1)
        cv2.rectangle(img, (h1 - t, h0), (h1, h1), (235, 235, 235, 255), -1)
        cv2.rectangle(img, (h0, c - t // 2), (h1, c + t // 2), (235, 235, 235, 255), -1)
    elif kind == "pad_red":
        cv2.circle(img, (c, c), c - 1, (30, 30, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), int(c * 0.55), (240, 240, 240, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), int(c * 0.30), (30, 30, 200, 255), -1, cv2.LINE_AA)
    elif kind == "pad_blue":
        m = max(1, px // 14)
        cv2.rectangle(img, (0, 0), (px - 1, px - 1), (190, 90, 20, 255), -1)
        cv2.rectangle(img, (m * 3, m * 3), (px - 1 - m * 3, px - 1 - m * 3), (240, 240, 240, 255), m)
        cv2.circle(img, (c, c), int(c * 0.25), (240, 240, 240, 255), -1, cv2.LINE_AA)
    elif kind == "cross_yellow":
        t = max(3, px // 4)
        cv2.rectangle(img, (c - t // 2, 0), (c + t // 2, px - 1), (20, 200, 235, 255), -1)
        cv2.rectangle(img, (0, c - t // 2), (px - 1, c + t // 2), (20, 200, 235, 255), -1)
    elif kind == "aruco":
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        quiet = max(2, px // 8)  # white quiet zone around the marker
        inner = px - 2 * quiet
        m = cv2.aruco.generateImageMarker(d, int(tag_id or 0), inner)
        img[:, :, :3] = 255
        img[quiet:quiet + inner, quiet:quiet + inner, :3] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        img[:, :, 3] = 255
    else:
        raise ValueError(f"unknown landmark type {kind}")
    return img


def random_distractor(px: int, rng: np.random.Generator) -> np.ndarray:
    """Painted shapes in colours / forms that are not landmark classes."""
    px = max(px, 12)
    img = np.zeros((px, px, 4), np.uint8)
    palette = [(40, 140, 40), (20, 110, 230), (150, 60, 150), (90, 160, 180), (220, 220, 220),
               (30, 30, 30), (60, 60, 140), (150, 120, 60)]
    col = palette[rng.integers(len(palette))] + (255,)
    kind = rng.integers(5)
    c = px // 2
    if kind == 0:
        cv2.rectangle(img, (0, px // 4), (px - 1, 3 * px // 4), col, -1)
    elif kind == 1:
        cv2.circle(img, (c, c), c - 1, col, max(2, px // 10), cv2.LINE_AA)
    elif kind == 2:
        pts = np.array([[c, 0], [px - 1, px - 1], [0, px - 1]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    elif kind == 3:
        for i in range(0, px, max(4, px // 6)):
            cv2.line(img, (i, 0), (i, px - 1), col, max(1, px // 16))
    else:
        cv2.ellipse(img, (c, c), (c - 1, c // 2), 0, 0, 360, col, -1, cv2.LINE_AA)
    return img


def paste(tex: np.ndarray, patch: np.ndarray, col: float, row: float, angle_deg: float) -> None:
    """Alpha-blend a BGRA patch onto the texture, centred at (col, row), rotated
    clockwise by ``angle_deg`` (i.e. a world yaw, since north is up)."""
    h, w = patch.shape[:2]
    diag = int(math.ceil(math.hypot(h, w))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    M[0, 2] += diag / 2 - w / 2
    M[1, 2] += diag / 2 - h / 2
    rot = cv2.warpAffine(patch, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    x0, y0 = int(round(col - diag / 2)), int(round(row - diag / 2))
    H, W = tex.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + diag), min(H, y0 + diag)
    if xa >= xb or ya >= yb:
        return
    sub = rot[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    region = tex[ya:yb, xa:xb].astype(np.float32)
    tex[ya:yb, xa:xb] = (a * sub[:, :, :3] + (1 - a) * region).astype(np.uint8)
