#!/usr/bin/env bash
# PX4 SITL helper (Ubuntu 22.04 / 24.04).
#
#   ./scripts/px4_sitl.sh setup            one time: clone PX4 v1.17.0 + toolchain + Gazebo Harmonic (~30 min)
#   ./scripts/px4_sitl.sh sih              fastest check, NO Gazebo: PX4's built-in physics (SIH, GPS on)
#   ./scripts/px4_sitl.sh run              Gazebo, stock x500 with optical flow + rangefinder (GPS off)
#   ./scripts/px4_sitl.sh gdnav [world]    Gazebo, our arena (landmarks, pillars) + downward camera
#   ./scripts/px4_sitl.sh headless ...     same as run/gdnav but without the Gazebo GUI (HEADLESS=1)
#
# Then, in a second terminal:
#   python scripts/apply_px4_params.py                         (skip for 'sih', which needs GPS)
#   python scripts/run_px4.py --mission hello_square           (sih / run)
#   python scripts/run_px4.py --mission investor_demo --gz-camera   (gdnav)
# QGroundControl connects automatically on UDP 14550. Full guide: docs/GAZEBO.md
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_TAG="${PX4_TAG:-v1.17.0}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

mode="${1:-run}"
if [ "$mode" = "headless" ]; then export HEADLESS=1; shift; mode="${1:-run}"; fi

case "$mode" in
  setup)
    if [ ! -d "$PX4_DIR" ]; then
      git clone --recursive --branch "$PX4_TAG" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
    fi
    bash "$PX4_DIR/Tools/setup/ubuntu.sh"   # toolchain + Gazebo Harmonic
    cd "$PX4_DIR" && make px4_sitl
    echo "PX4 built. Log out/in once (group changes), then: $0 run"
    ;;
  sih)
    cd "$PX4_DIR" && exec make px4_sitl sihsim_quadx
    ;;
  run)
    cd "$PX4_DIR" && exec make px4_sitl gz_x500_flow
    ;;
  gdnav)
    world="${2:-indoor_arena}"
    "$HERE/scripts/install_gazebo_assets.sh" "$PX4_DIR" "$world"
    cd "$PX4_DIR" && make px4_sitl >/dev/null
    cd "$PX4_DIR/build/px4_sitl_default/rootfs"
    # 4021 = x500_flow airframe (GPS disabled); our model adds the downward camera
    PX4_SYS_AUTOSTART=4021 PX4_SIM_MODEL=gz_x500_flow_cam_down PX4_GZ_WORLD="gdnav_${world}" \
      GZ_IP=127.0.0.1 exec ../bin/px4
    ;;
  *)
    echo "usage: $0 [headless] setup|sih|run|gdnav [world]"; exit 1 ;;
esac
