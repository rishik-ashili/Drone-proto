#!/usr/bin/env python3
"""Build an HTML + PNG flight report from a JSONL log.

    python scripts/make_report.py logs/<file>.jsonl [--out reports/run1]
    python scripts/make_report.py --latest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.common.config import LOGS_DIR  # noqa: E402
from gdnav.tools.report import build_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?")
    ap.add_argument("--latest", action="store_true", help="use the newest log in logs/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.latest or not a.log:
        logs = sorted(LOGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            sys.exit("no logs found in logs/")
        a.log = logs[-1]
    out = build_report(a.log, a.out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
