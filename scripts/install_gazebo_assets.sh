#!/usr/bin/env bash
# Copy the gdnav Gazebo world + x500_flow_cam_down model into a PX4-Autopilot checkout.
#   ./scripts/install_gazebo_assets.sh ~/PX4-Autopilot [world]
set -euo pipefail
cd "$(dirname "$0")/.."
PX4_DIR="${1:-$HOME/PX4-Autopilot}"
WORLD="${2:-indoor_arena}"
GZ="$PX4_DIR/Tools/simulation/gz"
[ -d "$GZ/models/x500_flow" ] || { echo "ERROR: $GZ/models/x500_flow not found (PX4 v1.17 checkout with submodules?)"; exit 1; }
PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"
$PY scripts/export_gazebo.py --world "$WORLD"
mkdir -p "$GZ/worlds/materials"
cp -r gazebo/models/x500_flow_cam_down "$GZ/models/"
cp "gazebo/worlds/materials/${WORLD}_floor.png" "$GZ/worlds/materials/"
# absolute texture path: Gazebo versions differ in how they resolve relative albedo maps
sed "s#<albedo_map>materials/#<albedo_map>file://$GZ/worlds/materials/#" \
    "gazebo/worlds/gdnav_${WORLD}.sdf" > "$GZ/worlds/gdnav_${WORLD}.sdf"
echo "Installed into $GZ:"
echo "  models/x500_flow_cam_down  worlds/gdnav_${WORLD}.sdf  worlds/materials/${WORLD}_floor.png"
echo
echo "Run (after 'make px4_sitl' has built PX4 once):"
echo "  ./scripts/px4_sitl.sh gdnav $WORLD"
