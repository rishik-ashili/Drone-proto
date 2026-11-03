#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source /opt/ros_ws/install/setup.bash
exec "$@"
