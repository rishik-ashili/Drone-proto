"""ROS-independent helpers for talking to PX4 over uXRCE-DDS (px4_msgs, PX4 v1.17).

Kept free of rclpy so it can be unit-tested anywhere. Values checked against the
PX4 v1.17.0 sources: msg/versioned/*.msg, src/modules/commander/px4_custom_mode.h
and src/modules/uxrce_dds_client (topic names get a ``_v<N>`` suffix when the
message has MESSAGE_VERSION > 0).
"""
from __future__ import annotations

import math

import numpy as np

# --- topics (PX4 v1.17 default dds_topics.yaml, no namespace)
TOPICS = {
    "local_position": "/fmu/out/vehicle_local_position_v1",   # VehicleLocalPosition MESSAGE_VERSION = 1
    "status": "/fmu/out/vehicle_status_v1",                   # VehicleStatus MESSAGE_VERSION = 1
    "battery": "/fmu/out/battery_status_v1",                  # BatteryStatus MESSAGE_VERSION = 1
    "attitude": "/fmu/out/vehicle_attitude",
    "land_detected": "/fmu/out/vehicle_land_detected",
    "offboard_mode": "/fmu/in/offboard_control_mode",
    "trajectory_setpoint": "/fmu/in/trajectory_setpoint",
    "vehicle_command": "/fmu/in/vehicle_command",
    "visual_odometry": "/fmu/in/vehicle_visual_odometry",
}

# --- VehicleCommand ids (msg/versioned/VehicleCommand.msg)
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

# --- PX4 custom modes (px4_custom_mode.h)
MAIN_POSCTL, MAIN_AUTO, MAIN_OFFBOARD = 3, 4, 6
SUB_AUTO_LOITER, SUB_AUTO_LAND = 3, 6

# --- VehicleStatus.nav_state -> names used by the mission state machine
NAV_STATE_NAMES = {0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "HOLD", 5: "RTL",
                   14: "OFFBOARD", 15: "STAB", 17: "AUTO_TAKEOFF", 18: "LAND", 20: "PRECLAND"}
ARMING_STATE_ARMED = 2
POSE_FRAME_NED = 1
VELOCITY_FRAME_NED = 1
VELOCITY_FRAME_BODY_FRD = 3


def set_mode_params(mode: str) -> tuple[float, float, float]:
    """(param1, param2, param3) for VEHICLE_CMD_DO_SET_MODE. param1 = 1: custom mode enabled."""
    table = {"offboard": (1.0, MAIN_OFFBOARD, 0.0), "posctl": (1.0, MAIN_POSCTL, 0.0),
             "hold": (1.0, MAIN_AUTO, SUB_AUTO_LOITER), "land": (1.0, MAIN_AUTO, SUB_AUTO_LAND)}
    return table[mode]


def nav_state_name(nav_state: int) -> str:
    """VehicleStatus.nav_state number -> mode name used by the state machine."""
    return NAV_STATE_NAMES.get(int(nav_state), f"NAV_{int(nav_state)}")


def quat_to_euler(q) -> np.ndarray:
    """PX4 quaternion [w, x, y, z] (FRD body -> NED) to roll, pitch, yaw."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll, pitch, yaw -> PX4 quaternion [w, x, y, z] (FRD body -> NED)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def trajectory_setpoint_fields(sp) -> dict:
    """gdnav Setpoint -> TrajectorySetpoint fields (NaN = not controlled) + matching OffboardControlMode."""
    nan = float("nan")
    if sp.kind == "position":
        position = [float(v) for v in sp.pos]
        velocity = [nan, nan, nan]
        mode = {"position": True, "velocity": False}
    else:
        position = [nan, nan, nan]
        velocity = [float(v) for v in sp.vel]
        mode = {"position": False, "velocity": True}
    yaw = float(sp.yaw) if sp.yaw is not None else nan
    yawspeed = nan if sp.yaw is not None else float(sp.yaw_rate)
    return {"position": position, "velocity": velocity, "acceleration": [nan] * 3, "jerk": [nan] * 3,
            "yaw": yaw, "yawspeed": yawspeed, "offboard_mode": mode}


def visual_odometry_fields(pos_xy, z: float, yaw: float, roll: float, pitch: float, vel_xy=None, vz: float = float("nan"),
                           pos_std: float = 0.05, yaw_std: float = 0.05, vel_std: float = 0.1,
                           reset_counter: int = 0) -> dict:
    """VehicleOdometry fields for /fmu/in/vehicle_visual_odometry (pose + velocity in NED)."""
    nan = float("nan")
    vel = [float(vel_xy[0]), float(vel_xy[1]), float(vz)] if vel_xy is not None else [nan] * 3
    return {"pose_frame": POSE_FRAME_NED,
            "position": [float(pos_xy[0]), float(pos_xy[1]), float(z)],
            "q": [float(v) for v in euler_to_quat(roll, pitch, yaw)],
            "velocity_frame": VELOCITY_FRAME_NED, "velocity": vel,
            "angular_velocity": [nan] * 3,
            "position_variance": [pos_std ** 2, pos_std ** 2, 0.05 ** 2],
            "orientation_variance": [0.05 ** 2, 0.05 ** 2, yaw_std ** 2],
            "velocity_variance": [vel_std ** 2] * 3,
            "reset_counter": int(reset_counter) % 256, "quality": 0}
