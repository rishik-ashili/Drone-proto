#!/usr/bin/env python3
"""Apply config/px4/gps_denied_flow.params to PX4 (SITL or Pixhawk) over MAVSDK.

    python scripts/apply_px4_params.py --url udp://:14540
    python scripts/apply_px4_params.py --url serial:///dev/ttyACM0:921600 --dry-run

Reboot the flight controller afterwards so EKF2 picks up the changes.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_params(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, value = line.split()[:2]
            out.append((name, value))
    return out


async def apply(url: str, params: list[tuple[str, str]]) -> None:
    try:
        from mavsdk_grpc import System
    except ImportError:
        from mavsdk import System
    drone = System()
    await drone.connect(system_address=url)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    for name, value in params:
        try:
            if "." in value:
                await drone.param.set_param_float(name, float(value))
            else:
                await drone.param.set_param_int(name, int(value))
            print(f"  set {name} = {value}")
        except Exception as e:  # noqa: BLE001 - report and continue (name may differ by version)
            print(f"  FAILED {name}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="udp://:14540")
    ap.add_argument("--file", default=str(ROOT / "config/px4/gps_denied_flow.params"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    params = read_params(Path(a.file))
    if a.dry_run:
        for n, v in params:
            print(f"{n} {v}")
        return
    asyncio.run(apply(a.url, params))


if __name__ == "__main__":
    main()
