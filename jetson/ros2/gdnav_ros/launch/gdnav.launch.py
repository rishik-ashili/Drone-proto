"""Jetson bring-up: Micro XRCE-DDS Agent (PX4 link) + gdnav mission node.

    ros2 launch gdnav_ros gdnav.launch.py mission:=investor_demo nav_source:=vio \\
        px4_port:=/dev/ttyTHS1 camera_topic:=/camera/down/image_raw

Start the stereo camera and Isaac ROS Visual SLAM separately (their launch files
depend on the camera model), e.g. for a RealSense D455:
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Micro XRCE-DDS Agent + mission node with launch arguments."""
    args = [
        DeclareLaunchArgument("mission", default_value="investor_demo"),
        DeclareLaunchArgument("nav_source", default_value="flow", description="flow | vio"),
        DeclareLaunchArgument("camera_topic", default_value="/camera/down/image_raw"),
        DeclareLaunchArgument("vio_topic", default_value="/visual_slam/tracking/odometry"),
        DeclareLaunchArgument("floor_map", default_value=""),
        DeclareLaunchArgument("autostart", default_value="false"),
        DeclareLaunchArgument("start_agent", default_value="true"),
        DeclareLaunchArgument("px4_port", default_value="/dev/ttyTHS1", description="Pixhawk TELEM2 UART"),
        DeclareLaunchArgument("px4_baud", default_value="921600"),
    ]
    agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "serial", "--dev", LaunchConfiguration("px4_port"), "-b", LaunchConfiguration("px4_baud")],
        output="screen", condition=IfCondition(LaunchConfiguration("start_agent")))
    mission = Node(
        package="gdnav_ros", executable="mission_node", name="gdnav_mission", output="screen",
        parameters=[{k: LaunchConfiguration(k) for k in
                     ("mission", "nav_source", "camera_topic", "vio_topic", "floor_map", "autostart")}])
    return LaunchDescription(args + [agent, mission])
