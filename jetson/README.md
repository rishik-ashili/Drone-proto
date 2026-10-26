# Phase 3: Jetson Orin migration

Target: Jetson Orin Nano 8GB / Orin NX, JetPack 6.x, ROS 2 Humble, a stereo + IMU camera (RealSense
D455 / OAK-D), PX4 over uXRCE-DDS.

## What is here

| File | Purpose |
|---|---|
| `vio_bridge_node.py` | ROS 2 node: Isaac ROS cuVSLAM `nav_msgs/Odometry` (ENU/FLU) → `px4_msgs/VehicleOdometry` on `/fmu/in/vehicle_visual_odometry` (NED/FRD) |
| `../gdnav/bridge/frames.py` | Frame and quaternion conversions used by the bridge; unit-tested in `tests/test_frames.py` |
| `../scripts/trt_engine_builder.py` | ONNX → TensorRT (FP16/INT8) with `trtexec`, plus a backend benchmark |

The perception, localizer, state machine and safety code (`gdnav/`) run unchanged on the Jetson. In
Phase 3, the `VehicleInterface` implementation moves from MAVSDK to ROS 2 topics. The bridge above is
its odometry half.

## Steps

1. Flash JetPack 6.x, set max power mode (`sudo nvpmodel -m 0 && sudo jetson_clocks`), and add
   active cooling and a separate regulated supply.
2. Install ROS 2 Humble, `px4_msgs` (matching your PX4 version) and the Micro XRCE-DDS Agent:
   `MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600`.
3. Calibrate the camera and IMU with Kalibr, then run Isaac ROS Visual SLAM (cuVSLAM) for your camera.
4. Run `python3 jetson/vio_bridge_node.py`.
5. In PX4, set `EKF2_EV_CTRL = 15` (position + velocity + yaw), keep `EKF2_GPS_CTRL = 0`, and tune
   `EKF2_EV_DELAY`.
6. Bench test with props off: carry the drone forward and check that PX4 local position x increases.
   Rotate it clockwise and check that yaw increases.
7. Build the detector engine: `python3 scripts/trt_engine_builder.py --fp16`. The ONNX Runtime
   TensorRT execution provider (`onnxruntime-gpu` for Jetson) then runs the same ONNX file.
8. Measure the targets: detection at 30 FPS or better, odometry into PX4 at 30 Hz or better, and
   drift as a percentage of distance flown. Compare with the POC using `scripts/benchmark.py`.

This code has not been run on a Jetson in this repository's CI. Treat it as a starting point and
verify message definitions against the `px4_msgs` version you build.
