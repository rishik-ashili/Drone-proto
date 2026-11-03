#!/usr/bin/env python3
"""Print the arena landmarks at their real size (PDF), tiled over A4/A3 pages.

    python scripts/make_printables.py                    # every landmark of indoor_arena -> printables/
    python scripts/make_printables.py --world plan_arena --paper A3
    python scripts/make_printables.py --only tag_07 --size-m 0.5

The artwork is exactly what the simulator draws and the YOLO model was trained on
(gdnav/sim/world.py: render_landmark). Each PDF page carries crop marks, tile
numbers and a 10 cm scale bar: measure it after printing (printers often scale by
2-3 %, use "actual size"). Matte paper or vinyl: glossy prints reflect lights.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.sim.world import World, render_landmark  # noqa: E402

PAPER_MM = {"A4": (210, 297), "A3": (297, 420), "letter": (216, 279)}
MARGIN_MM = 10
DPI = 300


def landmark_image(kind: str, size_m: float, tag_id) -> np.ndarray:
    px = int(round(size_m * 1000 / 25.4 * DPI))
    patch = render_landmark(kind, px, tag_id)
    rgb = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    return (rgb * a + 255 * (1 - a)).astype(np.uint8)  # white outside the shape


def write_pdf(path: Path, title: str, img: np.ndarray, size_m: float, paper: str) -> int:
    pw, ph = PAPER_MM[paper]
    uw, uh = pw - 2 * MARGIN_MM, ph - 2 * MARGIN_MM - 12
    size_mm = size_m * 1000
    nx, ny = math.ceil(size_mm / uw), math.ceil(size_mm / uh)
    px_per_mm = img.shape[1] / size_mm
    pages = 0
    with PdfPages(path) as pdf:
        for j in range(ny):
            for i in range(nx):
                x0, y0 = i * uw, j * uh
                w_mm, h_mm = min(uw, size_mm - x0), min(uh, size_mm - y0)
                crop = img[int(y0 * px_per_mm):int((y0 + h_mm) * px_per_mm), int(x0 * px_per_mm):int((x0 + w_mm) * px_per_mm)]
                fig = plt.figure(figsize=(pw / 25.4, ph / 25.4))
                ax = fig.add_axes([MARGIN_MM / pw, 1 - (MARGIN_MM + h_mm) / ph, w_mm / pw, h_mm / ph])
                ax.imshow(crop, interpolation="nearest")
                ax.set_axis_off()
                fig.text(MARGIN_MM / pw, 1 - (MARGIN_MM - 4) / ph,
                         f"{title}  |  {size_m * 100:.0f} cm  |  tile {j * nx + i + 1}/{nx * ny} (row {j + 1}, col {i + 1})",
                         fontsize=8, va="top")
                bar = fig.add_axes([MARGIN_MM / pw, 8 / ph, 100 / pw, 2 / ph])   # 10 cm scale bar
                bar.set_facecolor("black")
                bar.set_xticks([])
                bar.set_yticks([])
                fig.text((MARGIN_MM + 102) / pw, 9 / ph, "10 cm: measure after printing", fontsize=7, va="center")
                for cx, cy in ((x0, y0), (x0 + w_mm, y0), (x0, y0 + h_mm), (x0 + w_mm, y0 + h_mm)):  # crop marks
                    fx = (MARGIN_MM + cx - x0) / pw
                    fy = 1 - (MARGIN_MM + cy - y0) / ph
                    fig.lines.append(plt.Line2D([fx - 0.015, fx + 0.015], [fy, fy], transform=fig.transFigure, lw=0.4, color="k"))
                    fig.lines.append(plt.Line2D([fx, fx], [fy - 0.01, fy + 0.01], transform=fig.transFigure, lw=0.4, color="k"))
                pdf.savefig(fig, dpi=DPI)
                plt.close(fig)
                pages += 1
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="indoor_arena")
    ap.add_argument("--paper", default="A4", choices=list(PAPER_MM))
    ap.add_argument("--only", default=None, help="one landmark name")
    ap.add_argument("--size-m", type=float, default=None, help="override the size from the world file")
    ap.add_argument("--out", default=str(ROOT / "printables"))
    a = ap.parse_args()
    w = World.load(a.world, build_texture=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for lm in w.landmarks:
        if a.only and lm.name != a.only:
            continue
        size = a.size_m or lm.size
        img = landmark_image(lm.type, size, lm.id)
        f = out / f"{w.name}_{lm.name}_{int(size * 100)}cm_{a.paper}.pdf"
        n = write_pdf(f, f"{lm.name} ({lm.type}{'' if lm.id is None else ' id ' + str(lm.id)})", img, size, a.paper)
        cv2.imwrite(str(f.with_suffix(".png")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{f.name}: {n} page(s)  |  place at NED ({lm.x}, {lm.y}) m, yaw {lm.yaw_deg} deg")


if __name__ == "__main__":
    main()
