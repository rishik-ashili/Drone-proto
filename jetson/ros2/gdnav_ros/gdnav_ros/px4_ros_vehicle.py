"""VehicleInterface over ROS 2 + PX4 uXRCE-DDS (Phase 3, Jetson Orin).

Same API as the simulator and the MAVSDK backend, so gdnav's MissionRunner,
perception and safety code run unchanged. Topic names and fields come from
gdnav/bridge/px4_ros_convert.py (checked against PX4 v1.17.0).

  out (PX4 -> us): vehicle_local_position_v1, vehicle_attitude, vehicle_status_v1,
                   vehicle_land_detected, battery_status_v1
  in  (us -> PX4): offboard_control_mode (heartbeat), trajectory_setpoint,
                   vehicle_command, vehicle_visual_odometry
  camera:          sensor_msgs/Image (downward camera)
  VIO:             nav_msgs/Odometry from Isaac ROS cuVSLAM (ENU / FLU)
"""
from __future__ import annotations

import threading
import time

import numpy as np
from nav_msgs.msg import Odometry
from px4_msgs.msg import (BatteryStatus, OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand,
                          VehicleLandDetected, VehicleLocalPosition, VehicleOdometry, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from gdnav.bridge import px4_ros_convert as px
from gdnav.bridge.frames import ros_odometry_to_px4, yaw_from_q
from gdnav.interface.vehicle import CameraFrame, Telemetry, VehicleInterface
from gdnav.sim.vio import VioSample

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class Px4RosVehicle(VehicleInterface):
    """VehicleInterface over ROS 2 px4_msgs (uXRCE-DDS), plus camera and cuVSLAM inputs."""
    def __init__(self, node: Node, camera_topic: str | None = "/camera/down/image_raw",
                 vio_topic: str | None = None, frame_size=(640, 480)):
        self.node = node
        self._lock = threading.Lock()
        self._telem = Telemetry()
        self._frame: CameraFrame | None = None
        self._vio: list[VioSample] = []
        self._sp = None
        self._frame_size = frame_size
        self._t0 = time.monotonic()
        self.sys_id = 1
        T = px.TOPICS
        node.create_subscription(VehicleLocalPosition, T["local_position"], self._on_local_position, PX4_QOS)
        node.create_subscription(VehicleAttitude, T["attitude"], self._on_attitude, PX4_QOS)
        node.create_subscription(VehicleStatus, T["status"], self._on_status, PX4_QOS)
        node.create_subscription(VehicleLandDetected, T["land_detected"], self._on_land, PX4_QOS)
        node.create_subscription(BatteryStatus, T["battery"], self._on_battery, PX4_QOS)
        self.pub_mode = node.create_publisher(OffboardControlMode, T["offboard_mode"], PX4_QOS)
        self.pub_sp = node.create_publisher(TrajectorySetpoint, T["trajectory_setpoint"], PX4_QOS)
        self.pub_cmd = node.create_publisher(VehicleCommand, T["vehicle_command"], PX4_QOS)
        self.pub_odom = node.create_publisher(VehicleOdometry, T["visual_odometry"], PX4_QOS)
        if camera_topic:
            node.create_subscription(Image, camera_topic, self._on_image, 2)
        if vio_topic:
            node.create_subscription(Odometry, vio_topic, self._on_vio, 10)
        # PX4 leaves Offboard if the OffboardControlMode heartbeat stops (> 2 Hz required)
        node.create_timer(0.05, self._heartbeat)

    # ----------------------------------------------------------- helpers
    def now(self) -> float:
        return time.monotonic() - self._t0

    def _px4_us(self) -> int:
        return int(self.node.get_clock().now().nanoseconds / 1000)

    def _upd(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._telem, k, v)
            self._telem.received_t = self._telem.t = self.now()

    def _command(self, command: int, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0) -> None:
        m = VehicleCommand()
        m.timestamp = self._px4_us()
        m.command = int(command)
        m.param1, m.param2, m.param3, m.param4 = float(p1), float(p2), float(p3), float(p4)
        m.param5, m.param6, m.param7 = float(p5), float(p6), float(p7)
        m.target_system, m.target_component = self.sys_id, 1
        m.source_system, m.source_component = 1, 1
        m.from_external = True
        self.pub_cmd.publish(m)

    # ------------------------------------------------------- subscribers
    def _on_local_position(self, m: VehicleLocalPosition) -> None:
        self._upd(pos=np.array([m.x, m.y, m.z]), vel=np.array([m.vx, m.vy, m.vz]),
                  pos_std=float(m.eph) if m.xy_valid else 99.0,
                  range_m=float(m.dist_bottom) if m.dist_bottom_valid else max(0.0, -float(m.z)))

    def _on_attitude(self, m: VehicleAttitude) -> None:
        self._upd(att=px.quat_to_euler(m.q))

    def _on_status(self, m: VehicleStatus) -> None:
        self.sys_id = int(m.system_id) if getattr(m, "system_id", 0) else self.sys_id
        self._upd(armed=m.arming_state == px.ARMING_STATE_ARMED, mode=px.nav_state_name(m.nav_state),
                  failsafe="failsafe" if m.failsafe else "")

    def _on_land(self, m: VehicleLandDetected) -> None:
        self._upd(landed=bool(m.landed))

    def _on_battery(self, m: BatteryStatus) -> None:
        if m.remaining >= 0:
            self._upd(battery=float(m.remaining), voltage=float(m.voltage_v))

    def _on_image(self, m: Image) -> None:
        import cv2
        buf = np.frombuffer(m.data, np.uint8)
        if m.encoding in ("mono8", "8UC1"):
            img = cv2.cvtColor(buf.reshape(m.height, m.width), cv2.COLOR_GRAY2BGR)
        else:
            img = buf.reshape(m.height, m.width, -1)[:, :, :3]
            if m.encoding.lower().startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (img.shape[1], img.shape[0]) != tuple(self._frame_size):
            img = cv2.resize(img, tuple(self._frame_size))
        tl = self.telemetry()
        self._frame = CameraFrame(self.now(), img, tl.att.copy(), tl.range_m, tl.pos.copy())

    def _on_vio(self, m: Odometry) -> None:
        p, o, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        pos, q, vel_frd = ros_odometry_to_px4([p.x, p.y, p.z], [o.w, o.x, o.y, o.z], [v.x, v.y, v.z])
        yaw = yaw_from_q(q)
        c, s = np.cos(yaw), np.sin(yaw)
        vel = np.array([c * vel_frd[0] - s * vel_frd[1], s * vel_frd[0] + c * vel_frd[1], vel_frd[2]])
        cov = np.array(m.pose.covariance).reshape(6, 6)
        std = float(np.sqrt(max(cov[0, 0], cov[1, 1], 1e-4)))
        with self._lock:
            self._vio.append(VioSample(self.now(), pos, vel, yaw, std, float(np.sqrt(max(cov[5, 5], 1e-4))), 0, 90))

    def _heartbeat(self) -> None:
        sp = self._sp
        m = OffboardControlMode()
        m.timestamp = self._px4_us()
        f = px.trajectory_setpoint_fields(sp) if sp is not None else {"offboard_mode": {"position": True, "velocity": False}}
        m.position = f["offboard_mode"]["position"]
        m.velocity = f["offboard_mode"]["velocity"]
        m.acceleration = m.attitude = m.body_rate = m.thrust_and_torque = m.direct_actuator = False
        self.pub_mode.publish(m)

    # ---------------------------------------------------- VehicleInterface
    def telemetry(self) -> Telemetry:
        with self._lock:
            return Telemetry(**self._telem.__dict__)

    def arm(self) -> bool:
        self._command(px.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        return True

    def start_offboard(self) -> bool:
        self._command(px.VEHICLE_CMD_DO_SET_MODE, *px.set_mode_params("offboard"))
        return True

    def send_setpoint(self, sp) -> None:
        self._sp = sp
        f = px.trajectory_setpoint_fields(sp)
        m = TrajectorySetpoint()
        m.timestamp = self._px4_us()
        m.position, m.velocity = f["position"], f["velocity"]
        m.acceleration, m.jerk = f["acceleration"], f["jerk"]
        m.yaw, m.yawspeed = f["yaw"], f["yawspeed"]
        self.pub_sp.publish(m)

    def land(self) -> None:
        self._command(px.VEHICLE_CMD_NAV_LAND)

    def hold(self) -> None:
        self._command(px.VEHICLE_CMD_DO_SET_MODE, *px.set_mode_params("hold"))

    def position_mode(self) -> bool:
        self._command(px.VEHICLE_CMD_DO_SET_MODE, *px.set_mode_params("posctl"))
        return True

    def send_vision_position(self, t, pos_xy, std, yaw=None) -> bool:
        tl = self.telemetry()
        return self.send_odometry(t, pos_xy, float(yaw if yaw is not None else tl.att[2]), None, std,
                                  0.05 if yaw is not None else 1.0, 1.0)

    def send_odometry(self, t, pos_xy, yaw, vel_xy, pos_std, yaw_std, vel_std, reset_counter=0) -> bool:
        tl = self.telemetry()
        f = px.visual_odometry_fields(pos_xy, float(tl.pos[2]), float(yaw), float(tl.att[0]), float(tl.att[1]),
                                      vel_xy, float(tl.vel[2]), pos_std, yaw_std, vel_std, reset_counter)
        m = VehicleOdometry()
        m.timestamp = m.timestamp_sample = self._px4_us()
        for k, v in f.items():
            setattr(m, k, v)
        self.pub_odom.publish(m)
        return True  # EKF2 gates on the flight controller

    def vio_samples(self) -> list:
        with self._lock:
            out, self._vio = self._vio, []
        return out

    def latest_frame(self) -> CameraFrame | None:
        return self._frame
