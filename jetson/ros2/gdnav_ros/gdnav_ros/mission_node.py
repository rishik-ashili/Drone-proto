"""mission_node: gdnav's companion main loop (perception -> estimator -> health ->
state machine) as a ROS 2 node driving PX4 over uXRCE-DDS.

    ros2 run gdnav_ros mission_node --ros-args -p mission:=investor_demo -p nav_source:=vio

Services (std_srvs/Trigger):  ~/start  ~/land  ~/take_control  ~/resume
Topic out (std_msgs/String):   ~/status   JSON snapshot at 2 Hz (state, faults, metrics)
"""
from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from gdnav.common.config import LOGS_DIR, load_sim_config
from gdnav.navigation.mission import Mission
from gdnav.runner import MissionRunner
from gdnav.sim.world import World
from gdnav_ros.px4_ros_vehicle import Px4RosVehicle


class MissionNode(Node):
    """ROS 2 node running MissionRunner on Px4RosVehicle, with Trigger services and a status topic."""
    def __init__(self):
        super().__init__("gdnav_mission")
        p = self.declare_parameter
        mission = p("mission", "investor_demo").value
        nav_source = p("nav_source", "flow").value
        camera_topic = p("camera_topic", "/camera/down/image_raw").value
        vio_topic = p("vio_topic", "/visual_slam/tracking/odometry").value
        floor_map = p("floor_map", "").value
        config = p("config", "").value
        autostart = p("autostart", False).value

        cfg = load_sim_config(config or None)
        cfg["nav_source"] = nav_source
        if floor_map:
            cfg["map_matching"]["map"] = floor_map
        m = Mission.load(mission)
        world = World.load(m.world, build_texture=False)  # landmark map only
        self.vehicle = Px4RosVehicle(self, camera_topic, vio_topic if nav_source == "vio" else None,
                                     (cfg["sensors"]["camera"]["width"], cfg["sensors"]["camera"]["height"]))
        LOGS_DIR.mkdir(exist_ok=True)
        log = LOGS_DIR / f"ros_{m.name}_{self.get_clock().now().nanoseconds // 1_000_000_000}.jsonl"
        self.runner = MissionRunner(cfg, m, world, self.vehicle, log, landmark_correction=True, annotate=False)
        self.get_logger().info(f"mission {m.name}, nav source {nav_source}, detector {self.runner.cv.detector_name}")

        self.create_timer(self.runner.dt, self._tick)
        self.status_pub = self.create_publisher(String, "~/status", 5)
        self.create_timer(0.5, self._status)
        for name, fn in (("start", self.runner.start), ("land", self.runner.land_now),
                         ("take_control", self.runner.take_control), ("resume", self.runner.resume)):
            self.create_service(Trigger, f"~/{name}", self._service(fn, name))
        self._printed = 0
        if autostart:
            self.runner.start()

    def _service(self, fn, name):
        def cb(_req, res):
            out = fn()
            res.success = out is not False
            res.message = f"{name}: {'ok' if res.success else 'refused'} (state {self.runner.sm.state.value})"
            return res
        return cb

    def _tick(self) -> None:
        self.runner.tick()
        new = min(self.runner.event_count - self._printed, len(self.runner.events))
        for ev in list(self.runner.events)[len(self.runner.events) - new:] if new > 0 else []:
            self.get_logger().info(f"[{ev['type']}] {ev['msg']}")
        self._printed = self.runner.event_count

    def _status(self) -> None:
        s = self.runner.snapshot()
        keep = {k: s[k] for k in ("t", "state", "mode", "armed", "battery", "faults", "wp_index", "metrics", "vio",
                                  "floor_map", "nav_source")}
        self.status_pub.publish(String(data=json.dumps(keep)))

    def destroy_node(self):
        """Close the runner (summary + log) before shutting down."""
        self.runner.close()
        super().destroy_node()


def main() -> None:
    """Entry point: spin the mission node; Ctrl+C commands a landing."""
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.runner.land_now()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
