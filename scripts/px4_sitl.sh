#!/usr/bin/env bash
# Phase 0: PX4 SITL with Gazebo, using the quadcopter that carries an optical
# flow sensor and a rangefinder (no GPS needed). Ubuntu 22.04 recommended.
#
#   ./scripts/px4_sitl.sh setup     # one time: clone PX4 + install toolchain (~20-40 min)
#   ./scripts/px4_sitl.sh run       # start SITL (gz_x500_flow), then in another terminal:
#   python scripts/apply_px4_params.py && python scripts/run_px4.py --mission hello_square
#
# QGroundControl (optional) connects automatically on UDP 14550.
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_TAG="${PX4_TAG:-v1.17.0}"
MODEL="${PX4_MODEL:-gz_x500_flow}"

case "${1:-run}" in
  setup)
    if [ ! -d "$PX4_DIR" ]; then
      git clone --recursive --branch "$PX4_TAG" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
    fi
    bash "$PX4_DIR/Tools/setup/ubuntu.sh"
    echo "PX4 toolchain installed. Log out/in (or reboot) once, then run: $0 run"
    ;;
  run)
    cd "$PX4_DIR"
    # If your PX4 version has no x500_flow model, use PX4_MODEL=gz_x500 (GPS on) for the
    # Phase-0 Offboard interface test only.
    exec make px4_sitl "$MODEL"
    ;;
  headless)
    cd "$PX4_DIR"
    HEADLESS=1 exec make px4_sitl "$MODEL"
    ;;
  *)
    echo "usage: $0 [setup|run|headless]"; exit 1 ;;
esac
