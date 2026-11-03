"""PX4 uXRCE-DDS conversions (no ROS needed) and a smoke test of the ROS 2 vehicle with stub modules."""
import math
import sys
import types

import numpy as np

from gdnav.bridge import px4_ros_convert as px
from gdnav.sim.flight_controller import Setpoint


def test_versioned_topic_names_match_px4_1_17():
    assert px.TOPICS["local_position"].endswith("_v1")
    assert px.TOPICS["status"].endswith("_v1")
    assert px.TOPICS["attitude"] == "/fmu/out/vehicle_attitude"


def test_mode_commands():
    assert px.set_mode_params("offboard") == (1.0, 6, 0.0)
    assert px.set_mode_params("hold") == (1.0, 4, 3)
    assert px.nav_state_name(14) == "OFFBOARD" and px.nav_state_name(18) == "LAND"


def test_quaternion_roundtrip():
    e = np.array([0.1, -0.05, 2.0])
    assert np.allclose(px.quat_to_euler(px.euler_to_quat(*e)), e, atol=1e-9)


def test_setpoint_fields_use_nan_for_uncontrolled_axes():
    f = px.trajectory_setpoint_fields(Setpoint("velocity", vel=np.array([1.0, 0.0, -0.2]), yaw=0.5))
    assert f["offboard_mode"] == {"position": False, "velocity": True}
    assert all(math.isnan(v) for v in f["position"]) and f["velocity"][0] == 1.0


def _install_stubs():
    """Minimal stand-ins for rclpy / px4_msgs / ROS message packages."""
    class Msg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def msg_module(name, classes):
        m = types.ModuleType(name)
        for c in classes:
            setattr(m, c, type(c, (Msg,), {}))
        return m

    px4 = msg_module("px4_msgs.msg", ["BatteryStatus", "OffboardControlMode", "TrajectorySetpoint", "VehicleAttitude",
                                      "VehicleCommand", "VehicleLandDetected", "VehicleLocalPosition",
                                      "VehicleOdometry", "VehicleStatus"])
    published = []

    class Pub:
        def __init__(self, t):
            self.t = t

        def publish(self, m):
            published.append((self.t, m))

    class Node:
        def __init__(self, *a, **k):
            pass

        def create_subscription(self, *a, **k):
            return None

        def create_publisher(self, t, topic, qos):
            return Pub(topic)

        def create_timer(self, *a, **k):
            return None

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1_000_000))

    qos = types.ModuleType("rclpy.qos")
    for n in ("DurabilityPolicy", "HistoryPolicy", "ReliabilityPolicy"):
        setattr(qos, n, types.SimpleNamespace(BEST_EFFORT=0, TRANSIENT_LOCAL=0, KEEP_LAST=0))
    qos.QoSProfile = lambda **k: k
    mods = {"rclpy": types.ModuleType("rclpy"), "rclpy.node": types.SimpleNamespace(Node=Node), "rclpy.qos": qos,
            "px4_msgs": types.ModuleType("px4_msgs"), "px4_msgs.msg": px4,
            "nav_msgs": types.ModuleType("nav_msgs"), "nav_msgs.msg": msg_module("nav_msgs.msg", ["Odometry"]),
            "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": msg_module("sensor_msgs.msg", ["Image"])}
    sys.modules.update(mods)
    return px4, published, Node


def test_ros_vehicle_smoke():
    import importlib
    from pathlib import Path

    px4, published, Node = _install_stubs()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson" / "ros2" / "gdnav_ros"))
    mod = importlib.import_module("gdnav_ros.px4_ros_vehicle")
    v = mod.Px4RosVehicle(Node())
    v._on_local_position(px4.VehicleLocalPosition(x=1.0, y=2.0, z=-2.0, vx=0.1, vy=0.0, vz=0.0, eph=0.05,
                                                  xy_valid=True, dist_bottom=2.0, dist_bottom_valid=True))
    v._on_attitude(px4.VehicleAttitude(q=[1.0, 0.0, 0.0, 0.0]))
    v._on_status(px4.VehicleStatus(arming_state=2, nav_state=14, failsafe=False, system_id=1))
    t = v.telemetry()
    assert t.mode == "OFFBOARD" and t.armed and np.allclose(t.pos, [1, 2, -2])
    v.send_setpoint(Setpoint("position", pos=np.array([3.0, 0.0, -2.5]), yaw=0.0))
    v._heartbeat()
    v.arm()
    v.send_odometry(0.0, np.array([1.0, 2.0]), 0.1, np.array([0.2, 0.0]), 0.05, 0.02, 0.1, reset_counter=2)
    topics = [tp for tp, _ in published]
    assert px.TOPICS["trajectory_setpoint"] in topics and px.TOPICS["offboard_mode"] in topics
    odom = [m for tp, m in published if tp == px.TOPICS["visual_odometry"]][0]
    assert odom.pose_frame == px.POSE_FRAME_NED and odom.reset_counter == 2
    cmd = [m for tp, m in published if tp == px.TOPICS["vehicle_command"]][0]
    assert cmd.command == px.VEHICLE_CMD_COMPONENT_ARM_DISARM and cmd.param1 == 1.0
