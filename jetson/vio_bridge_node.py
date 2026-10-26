#!/usr/bin/env python3
"""vio_bridge (Phase 3, Jetson Orin, ROS 2 Humble): Isaac ROS cuVSLAM odometry
-> PX4 external vision over uXRCE-DDS.

  subscribes  /visual_slam/tracking/odometry      nav_msgs/Odometry (ENU / FLU)
  publishes   /fmu/in/vehicle_visual_odometry     px4_msgs/VehicleOdometry (NED / FRD)

PX4 side: EKF2_EV_CTRL = 15 (pos + vel + yaw), EKF2_EV_DELAY tuned to the
measured camera->odometry latency. Bench-test with props off: walk the drone
forward and check that PX4's local position x increases (plan, section 11).

Run:  ros2 run <your_pkg> vio_bridge_node   (or: python3 jetson/vio_bridge_node.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.bridge.frames import ros_odometry_to_px4  # noqa: E402


class VioBridge(Node):
    def __init__(self):
        super().__init__("vio_bridge")
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("pos_variance_floor", 0.0004)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", qos)
        self.sub = self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.on_odom, 10)
        self.count = 0
        self.create_timer(5.0, self.report)

    def on_odom(self, m: Odometry) -> None:
        p, o, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        pos, q, vel = ros_odometry_to_px4([p.x, p.y, p.z], [o.w, o.x, o.y, o.z], [v.x, v.y, v.z])
        out = VehicleOdometry()
        stamp_us = m.header.stamp.sec * 1_000_000 + m.header.stamp.nanosec // 1000
        out.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        out.timestamp_sample = int(stamp_us)
        out.pose_frame = VehicleOdometry.POSE_FRAME_NED
        out.position = pos.astype(np.float32).tolist()
        out.q = q.astype(np.float32).tolist()
        out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        out.velocity = vel.astype(np.float32).tolist()
        out.angular_velocity = [float("nan")] * 3
        floor = self.get_parameter("pos_variance_floor").value
        cov = np.array(m.pose.covariance).reshape(6, 6)
        # ENU -> NED swaps x/y variances
        out.position_variance = [max(cov[1, 1], floor), max(cov[0, 0], floor), max(cov[2, 2], floor)]
        out.orientation_variance = [max(cov[4, 4], 1e-4), max(cov[3, 3], 1e-4), max(cov[5, 5], 1e-4)]
        tcov = np.array(m.twist.covariance).reshape(6, 6)
        out.velocity_variance = [max(tcov[0, 0], 1e-3), max(tcov[1, 1], 1e-3), max(tcov[2, 2], 1e-3)]
        out.reset_counter = 0
        out.quality = 100
        self.pub.publish(out)
        self.count += 1

    def report(self) -> None:
        self.get_logger().info(f"forwarded {self.count / 5.0:.1f} Hz odometry to PX4")
        self.count = 0


def main() -> None:
    rclpy.init()
    rclpy.spin(VioBridge())


if __name__ == "__main__":
    main()
