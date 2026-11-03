#!/usr/bin/env python3
"""Analyse a PX4 flight log (.ulg) for GPS-denied flight: what did EKF2 fuse, how good
was the position estimate, did the landmark / VIO fixes get through, which failsafes fired.

    python scripts/analyze_ulog.py log_12_2026-11-05-10-12-01.ulg       # -> <log>_report/report.html

Pull the log from the Pixhawk SD card (or QGroundControl > Analyze > Log Download).
Topics are optional: whatever the log contains is reported (older logs carry less).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAV_STATES = {0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "MISSION", 4: "HOLD", 5: "RTL", 14: "OFFBOARD",
              15: "STAB", 17: "TAKEOFF", 18: "LAND", 20: "PRECLAND"}
AID_FLAGS = ["cs_opt_flow", "cs_ev_pos", "cs_ev_vel", "cs_ev_yaw", "cs_gps", "cs_rng_hgt", "cs_baro_hgt", "cs_mag_hdg",
             "cs_inertial_dead_reckoning", "cs_wind_dead_reckoning"]


def load(path: str):
    from pyulog import ULog
    u = ULog(path)
    topics = {}
    for d in u.data_list:
        if d.multi_id == 0:
            topics[d.name] = d.data
    return u, topics


def tsec(d, t0):
    return (d["timestamp"].astype(np.float64) - float(t0)) / 1e6  # float: samples can precede the log start


def fig64(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def analyse(path: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    u, T = load(path)
    t0 = u.start_timestamp
    rows: list[tuple[str, str]] = [("log", Path(path).name),
                                   ("duration", f"{(u.last_timestamp - t0) / 1e6:.0f} s"),
                                   ("firmware", str(u.msg_info_dict.get("ver_sw_release", u.msg_info_dict.get("ver_sw", "?"))))]
    figs: dict[str, str] = {}
    p = {k: v for k, v in u.initial_parameters.items() if k.startswith(("EKF2_GPS_CTRL", "EKF2_OF_CTRL", "EKF2_EV_CTRL",
                                                                         "EKF2_HGT_REF", "COM_OF_LOSS_T", "COM_OBL_RC_ACT"))}
    rows += [(f"param {k}", str(v)) for k, v in sorted(p.items())]

    if "vehicle_local_position" in T:
        lp = T["vehicle_local_position"]
        t = tsec(lp, t0)
        rows.append(("local position valid", f"{100 * float(np.mean(lp['xy_valid'])):.0f} % of samples"))
        if "eph" in lp:
            rows.append(("position accuracy eph", f"median {np.median(lp['eph']):.2f} m, max {np.max(lp['eph']):.2f} m"))
        rows.append(("xy resets", str(int(lp["xy_reset_counter"].max() - lp["xy_reset_counter"].min()))))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(lp["y"], lp["x"], lw=1)
        ax[0].set_aspect("equal")
        ax[0].set_xlabel("east (m)")
        ax[0].set_ylabel("north (m)")
        ax[0].set_title("local position (estimate)")
        ax[1].plot(t, lp["eph"] if "eph" in lp else np.zeros_like(t), label="eph (m)")
        if "dist_bottom" in lp:
            ax[1].plot(t, lp["dist_bottom"], label="dist_bottom (m)")
        ax[1].legend()
        ax[1].set_xlabel("time (s)")
        ax[1].grid(alpha=.3)
        figs["local_position"] = fig64(fig)

    if "estimator_status_flags" in T:
        f = T["estimator_status_flags"]
        t = tsec(f, t0)
        present = [k for k in AID_FLAGS if k in f]
        rows += [(f"EKF2 {k}", f"{100 * float(np.mean(f[k])):.0f} % of flight") for k in present]
        fig, ax = plt.subplots(figsize=(11, 0.35 * len(present) + 1))
        for i, k in enumerate(present):
            ax.fill_between(t, i, i + 0.8 * f[k], step="post")
        ax.set_yticks(np.arange(len(present)) + 0.4)
        ax.set_yticklabels(present, fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_title("EKF2 aiding sources")
        figs["aiding"] = fig64(fig)
        if "cs_gps" in f and np.any(f["cs_gps"]):
            rows.append(("WARNING", "GPS was fused: not a GPS-denied flight (EKF2_GPS_CTRL)"))

    for name, label in (("estimator_aid_src_optical_flow", "optical flow"), ("estimator_aid_src_ev_pos", "external vision pos"),
                        ("estimator_aid_src_ev_vel", "external vision vel")):
        if name in T:
            a = T[name]
            fused = a.get("fused")
            tr = [a[k] for k in a if k.startswith("test_ratio")]
            if fused is not None:
                rows.append((f"{label} fused", f"{100 * float(np.mean(fused)):.0f} % of samples"))
            if tr:
                rows.append((f"{label} test ratio", f"median {np.median(np.max(tr, axis=0)):.2f} (>1 = rejected)"))
    if "sensor_optical_flow" in T and "quality" in T["sensor_optical_flow"]:
        q = T["sensor_optical_flow"]["quality"]
        rows.append(("flow quality", f"median {np.median(q):.0f} / 255, below 60: {100 * float(np.mean(q < 60)):.0f} %"))

    if "vehicle_status" in T:
        s = T["vehicle_status"]
        t = tsec(s, t0)
        ch = np.flatnonzero(np.diff(s["nav_state"])) + 1
        seq = [f"{t[0]:.0f}s {NAV_STATES.get(int(s['nav_state'][0]), s['nav_state'][0])}"]
        seq += [f"{t[i]:.0f}s {NAV_STATES.get(int(s['nav_state'][i]), int(s['nav_state'][i]))}" for i in ch]
        rows.append(("mode sequence", " -> ".join(seq[:25])))
        if "failsafe" in s and np.any(s["failsafe"]):
            first = t[np.argmax(s["failsafe"] > 0)]
            rows.append(("failsafe", f"active {100 * float(np.mean(s['failsafe'])):.0f} % of the log, first at {first:.0f} s"))
    if "failsafe_flags" in T:
        ff = T["failsafe_flags"]
        hit = [k for k in ff if k != "timestamp" and ff[k].dtype == np.uint8 and np.any(ff[k])]
        if hit:
            rows.append(("failsafe flags raised", ", ".join(hit[:12])))
    if "battery_status" in T:
        b = T["battery_status"]
        rows.append(("battery", f"{b['voltage_v'][0]:.2f} V -> {b['voltage_v'][-1]:.2f} V"))
    return rows, figs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ulog")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, figs = analyse(a.ulog)
    out = Path(a.out) if a.out else Path(a.ulog).with_name(Path(a.ulog).stem + "_report")
    out.mkdir(parents=True, exist_ok=True)
    for k, v in rows:
        print(f"  {k:32s} {v}")
    table = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    imgs = "".join(f'<h2>{n}</h2><img src="data:image/png;base64,{b}">' for n, b in figs.items())
    (out / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>ULog report</title><style>body{font-family:system-ui;max-width:1100px;"
        "margin:24px auto;padding:0 16px}td{padding:3px 10px;border-bottom:1px solid #ddd}img{max-width:100%}</style>"
        f"<h1>PX4 log: {html.escape(Path(a.ulog).name)}</h1><table>{table}</table>{imgs}")
    print(f"\nReport: {out / 'report.html'}")


if __name__ == "__main__":
    sys.exit(main())
