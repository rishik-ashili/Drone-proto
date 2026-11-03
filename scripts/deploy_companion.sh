#!/usr/bin/env bash
# Install gdnav on the companion computer (Raspberry Pi 4/5 or Jetson, 64-bit OS) as a boot-time service.
#   sudo ./scripts/deploy_companion.sh            # MAVSDK companion (Pi / Jetson without ROS)
#   sudo ./scripts/deploy_companion.sh --ros      # Jetson with ROS 2 (see jetson/README.md first)
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST=/opt/gdnav
id gdnav >/dev/null 2>&1 || useradd --system --create-home --groups dialout,video gdnav
mkdir -p "$DST" /etc/gdnav
rsync -a --delete --exclude .venv --exclude datasets --exclude runs --exclude logs "$SRC/" "$DST/"
python3 -m venv "$DST/.venv"
"$DST/.venv/bin/pip" install -U pip
"$DST/.venv/bin/pip" install -r "$DST/requirements-px4.txt"
mkdir -p "$DST/logs" && chown -R gdnav:gdnav "$DST"
[ -f /etc/gdnav/companion.env ] || cp "$DST/deploy/companion.env.example" /etc/gdnav/companion.env
unit=gdnav-companion.service
[ "${1:-}" = "--ros" ] && unit=gdnav-ros.service
cp "$DST/deploy/systemd/$unit" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now "$unit"
echo "Installed $unit. Edit /etc/gdnav/companion.env, then: sudo systemctl restart $unit; journalctl -fu $unit"
echo "Perception speed on this board: $DST/.venv/bin/python $DST/scripts/benchmark_perception.py"
