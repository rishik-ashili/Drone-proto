"""Flight report from a JSONL log: trajectory plot, error-over-time plot,
state timeline and a metrics table, written as PNGs + a self-contained HTML.

    python scripts/make_report.py logs/investor_demo_20261026_101500.jsonl
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gdnav.common.logger import read_log  # noqa: E402

C_TRUE, C_EST, C_DR = "#2fbf71", "#2a8fd6", "#f08a24"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_report(log_path: str | Path, out_dir: str | Path | None = None) -> Path:
    log_path = Path(log_path)
    data = read_log(log_path)
    meta = data["meta"][0] if data["meta"] else {}
    ticks = data["tick"]
    summary = data["summary"][-1] if data["summary"] else {}
    out_dir = Path(out_dir) if out_dir else log_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.array([k["t"] for k in ticks])
    est = np.array([k["est_pos"] for k in ticks])
    has_truth = "true_pos" in (ticks[0] if ticks else {})
    figs: dict[str, str] = {}

    # ---- trajectory
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    w = meta.get("world", {})
    if w:
        xmin, xmax, ymin, ymax = w["bounds"]
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(xmin, xmax)
        for o in w.get("obstacles", []):
            ax.add_patch(plt.Circle((o["y"], o["x"]), o["radius"], color="#777", alpha=0.8))
        for lm in w.get("landmarks", []):
            ax.plot(lm["y"], lm["x"], "s", color="#999", ms=5)
            ax.annotate(lm["name"], (lm["y"], lm["x"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#666")
    m = meta.get("mission", {})
    route = [(0, 0)] + [(wp["x"], wp["y"]) for wp in m.get("waypoints", [])]
    if m.get("on_finish") == "return_to_start":
        route.append((0, 0))
    r = np.array(route)
    ax.plot(r[:, 1], r[:, 0], "--", color="#bbb", lw=1, label="planned route")
    if has_truth:
        tru = np.array([k["true_pos"] for k in ticks])
        dr = np.array([k["dr_pos"] for k in ticks])
        ax.plot(dr[:, 1], dr[:, 0], color=C_DR, lw=1.2, label="dead reckoning only")
        ax.plot(tru[:, 1], tru[:, 0], color=C_TRUE, lw=2, label="true path")
    ax.plot(est[:, 1], est[:, 0], color=C_EST, lw=1.2, label="onboard estimate")
    for wp in m.get("waypoints", []):
        ax.plot(wp["y"], wp["x"], "o", mfc="white", mec="k", ms=9)
        ax.annotate(str(wp["id"]), (wp["y"], wp["x"]), ha="center", va="center", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Trajectory (local NED frame, no GPS)")
    figs["trajectory"] = _fig_to_b64(fig)

    # ---- errors over time
    if has_truth:
        e_est = np.linalg.norm(tru[:, :2] - est[:, :2], axis=1)
        e_dr = np.linalg.norm(tru[:, :2] - dr[:, :2], axis=1)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, e_dr, color=C_DR, label="dead reckoning (flow + IMU)")
        ax.plot(t, e_est, color=C_EST, label="with landmark correction")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("horizontal error (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Position estimate error vs. ground truth")
        figs["errors"] = _fig_to_b64(fig)

    # ---- state timeline
    states = [k["state"] for k in ticks]
    order = list(dict.fromkeys(states))
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(order) + 0.8))
    yi = np.array([order.index(s) for s in states])
    ax.scatter(t, yi, s=2, c=yi, cmap="tab10")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.3)
    ax.set_title("Mission state machine")
    figs["states"] = _fig_to_b64(fig)

    for name, b64 in figs.items():
        (out_dir / f"{name}.png").write_bytes(base64.b64decode(b64))

    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v))}</td></tr>"
                   for k, v in summary.items() if k not in ("waypoints", "kind", "wall"))
    wps = "".join(f"<tr><td>{w.get('id')}</td><td>{w.get('landmark') or '-'}</td><td>{'yes' if w.get('found') else 'no'}</td>"
                  f"<td>{w.get('reach_error_m', '-')}</td><td>{w.get('est_error_m', '-')}</td></tr>"
                  for w in summary.get("waypoints", []))
    events = "".join(f"<tr><td>{e['t']:.2f}</td><td>{html.escape(e['type'])}</td><td>{html.escape(e['msg'])}</td></tr>"
                     for e in data["event"] if e["type"] not in ("px4",))
    imgs = "".join(f'<img alt="{n}" src="data:image/png;base64,{b}">' for n, b in figs.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Flight report - {html.escape(m.get('name', ''))}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2733}}
img{{max-width:100%;margin:8px 0;border:1px solid #e3e8ee;border-radius:8px}}table{{border-collapse:collapse;font-size:13px;margin:8px 0}}
td,th{{border-bottom:1px solid #e3e8ee;padding:4px 10px;text-align:left}}h1{{font-size:22px}}h2{{font-size:16px;margin-top:28px}}</style></head><body>
<h1>Flight report · {html.escape(m.get('name', ''))}</h1>
<p>Log: <code>{html.escape(log_path.name)}</code> · backend {html.escape(str(meta.get('backend')))} · detector {html.escape(str(meta.get('detector')))} ·
landmark correction {'ON' if meta.get('landmark_correction') else 'OFF'}</p>
<h2>Metrics</h2><table>{rows}</table>
<h2>Waypoints</h2><table><tr><th>WP</th><th>Landmark</th><th>Found</th><th>True reach error (m)</th><th>Estimate error (m)</th></tr>{wps}</table>
<h2>Plots</h2>{imgs}
<h2>Events</h2><table>{events}</table></body></html>"""
    out = out_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
