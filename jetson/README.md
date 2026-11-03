# Phase 3: Jetson Orin migration

Target: Jetson Orin Nano 8GB or Orin NX, JetPack 6.x, ROS 2 Humble, a stereo + IMU camera (RealSense
D455 / OAK-D), and a Pixhawk running PX4 v1.17 over uXRCE-DDS on TELEM2.

## What is here

| Path | Purpose |
|---|---|
| `ros2/gdnav_ros/gdnav_ros/px4_ros_vehicle.py` | `VehicleInterface` over px4_msgs: the same mission, perception and safety code as the simulator, driving PX4 through `/fmu/in/*` |
| `ros2/gdnav_ros/gdnav_ros/mission_node.py` | The companion main loop as a ROS 2 node. Services `~/start`, `~/land`, `~/take_control`, `~/resume`; status JSON on `~/status` |
| `ros2/gdnav_ros/gdnav_ros/vio_bridge_node.py` | Standalone cuVSLAM → PX4 odometry bridge (ENU/FLU → NED/FRD) |
| `ros2/gdnav_ros/launch/gdnav.launch.py` | Micro XRCE-DDS Agent + mission node |
| `../gdnav/bridge/px4_ros_convert.py` | Topic names, command codes and message fields checked against the PX4 v1.17.0 source, unit-tested |
| `../gdnav/bridge/landmark_vio_fusion.py` | Landmark (and floor-map) correction of the VIO frame, so EKF2 gets one consistent EV stream |
| `../docker/Dockerfile.jetson`, `../docker/docker-compose.jetson.yml` | Container with the agent, px4_msgs `release/1.17`, gdnav and the ROS package |
| `../scripts/trt_engine_builder.py` | ONNX → TensorRT (FP16/INT8) plus a backend benchmark |

With `nav_source:=vio`, the mission node subscribes to cuVSLAM odometry. It corrects the odometry
with every confirmed landmark fix and streams the result to `/fmu/in/vehicle_visual_odometry`
(EKF2_EV_CTRL = 15). Optical flow stays fused as the fallback. This is the same logic that runs in the
simulator (`python scripts/run_sim.py --nav-source vio`).

PX4 v1.17 suffixes versioned topics: `vehicle_status_v1`, `vehicle_local_position_v1`,
`battery_status_v1`. Older PX4 examples use the unsuffixed names and would receive nothing.

## Steps on the Jetson

1. Flash JetPack 6.x. Set max power mode (`sudo nvpmodel -m 0 && sudo jetson_clocks`), and fit
   active cooling and a separate regulated supply.
2. On the Pixhawk: `UXRCE_DDS_CFG = 102` (TELEM2), `SER_TEL2_BAUD = 921600`, then apply
   `config/px4/gps_denied_flow.params` with `EKF2_EV_CTRL = 15` for VIO. Wire TELEM2 to
   `/dev/ttyTHS1` (Orin UART).
3. Calibrate the stereo camera and IMU with Kalibr. Run Isaac ROS Visual SLAM for your camera and
   check `/visual_slam/tracking/odometry`.
4. Either use Docker:
   ```bash
   docker build -f docker/Dockerfile.jetson --build-arg BASE_IMAGE=<ros2 humble image for your L4T> -t gdnav-jetson .
   docker compose -f docker/docker-compose.jetson.yml up
   ```
   or build natively:
   ```bash
   pip install -e .                      # gdnav
   mkdir -p ~/ros_ws/src && cd ~/ros_ws/src
   git clone -b release/1.17 https://github.com/PX4/px4_msgs.git
   ln -s ~/Drone-proto/jetson/ros2/gdnav_ros .
   cd ~/ros_ws && colcon build && source install/setup.bash
   ros2 launch gdnav_ros gdnav.launch.py nav_source:=vio
   ```
5. Bench test with props **off**: `ros2 service call /gdnav_mission/start std_srvs/srv/Trigger` and
   watch the state machine. Carry the drone forward and check that PX4's local position x increases.
6. Build the detector engine: `python3 scripts/trt_engine_builder.py --fp16`, then measure with
   `python3 scripts/benchmark_perception.py`. Targets: detection at 30 FPS or better, odometry into
   PX4 at 30 Hz or better.

## Not yet verified

None of this has run on a Jetson or with real ROS 2 yet. The conversions and the vehicle class are
tested in `tests/test_ros_bridge.py` with stand-in ROS modules, and the message fields were checked
against the PX4 v1.17.0 `.msg` files. Expect small fixes on first contact. Send the error output.
