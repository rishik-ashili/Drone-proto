# Architecture: how the plan maps to code

The layers and modules below are taken from sections 3 and 7 of the phase plan.

## Layers

| Layer (plan) | In the simulator | On the real drone |
|---|---|---|
| L1 Flight control + state estimation | `gdnav/sim/flight_controller.py` (PX4 position/velocity cascade, offboard-loss and battery failsafes), `gdnav/sim/estimator.py` (EKF2 analogue: IMU + flow + rangefinder + baro + mag + external vision, chi² gating) | PX4 on the Pixhawk 6C, parameters in `config/px4/gps_denied_flow.params` |
| L2 Perception | `gdnav/perception/*` | same code |
| L3 Navigation + mission logic | `gdnav/navigation/*` | same code |
| L4 Interface | `gdnav/sim/sim_vehicle.py` implements `VehicleInterface` | `gdnav/interface/px4_mavsdk.py` (MAVSDK Offboard + Mocap); ROS 2 on Orin |
| L5 Safety + monitoring | `gdnav/safety/health_monitor.py`, `navigation/geofence.py` | same code + PX4 failsafes |
| L6 Tooling | `gdnav/dashboard`, `gdnav/tools`, `scripts/` | same, plus PX4 SITL / QGroundControl |

Data flow per navigation tick (20 Hz), in `gdnav/runner.py`:

```
camera frame ─► cv_node (YOLO + ArUco) ─► landmark_localizer ─► association ─► temporal filter
      │                                                                              │ confirmed
      │                                    send_vision_position (EKF2_EV_CTRL) ◄─────┤
      ▼                                                                              ▼
health_monitor ─► mission_state_machine (+ local_planner, geofence) ─► Offboard setpoint ─► PX4
```

## Modules from the plan's section 7

| Module | File | Notes |
|---|---|---|
| cv_node | `gdnav/perception/cv_node.py` | YOLO (ONNX Runtime / OpenCV DNN / TensorRT EP) + ArUco, publishes detections with confidence and timestamp |
| landmark_localizer | `gdnav/perception/landmark_localizer.py` | pixel + range + estimated attitude + intrinsics → metric offset; ArUco heading fix |
| temporal filter | `gdnav/perception/temporal_filter.py` | N consecutive frames before a landmark reaches navigation |
| waypoint_manager | `gdnav/navigation/mission.py` | loads mission YAML (section 9 format), tracks progress and per-waypoint results |
| mission_state_machine | `gdnav/navigation/state_machine.py` | TAKEOFF, GOTO_WAYPOINT, SEARCH_LANDMARK, ALIGN, NEXT_WAYPOINT, RETURN_TO_START, PRECISION_LAND, LAND, FAILSAFE_HOLD |
| px4_interface | `gdnav/interface/vehicle.py`, `px4_mavsdk.py` | one API for sim and PX4 |
| health_monitor | `gdnav/safety/health_monitor.py` | battery ladder, link, camera, position quality, geofence → hold / RTS / land |
| return_to_start | `state_machine._s_return_to_start` + `_s_precision_land` | Offboard, local frame (PX4 Return needs global position) |
| vio_bridge | `jetson/vio_bridge_node.py` + `gdnav/bridge/frames.py` | ENU/FLU → NED/FRD, tested in `tests/test_frames.py` |
| trt_engine_builder | `scripts/trt_engine_builder.py` | trtexec FP16/INT8 + backend benchmark |
| local_planner | `gdnav/navigation/local_planner.py` | reactive avoidance on a depth fan (EGO-Planner stand-in, same interface) |
| logger and replay | `gdnav/common/logger.py`, `gdnav/tools/replay.py`, `gdnav/tools/report.py` | JSONL; report HTML; replay in the dashboard |
| ground dashboard | `gdnav/dashboard/` | telemetry, detections, state machine, failure injection |

## Simulation models

| Model | File | Key parameters (`config/sim.yaml`) |
|---|---|---|
| Arena + floor texture | `gdnav/sim/world.py` | `worlds/*.yaml`: landmarks, obstacles, distractors, surface |
| Downward camera | `gdnav/sim/camera.py` | 640×480, 80° HFOV, homography render, vignetting, noise, motion blur |
| Quadrotor | `gdnav/sim/quadrotor.py` | 1.6 kg, thrust limit, drag, first-order attitude response, wind, battery |
| Sensors | `gdnav/sim/sensors.py` | IMU noise + bias random walk, flow scale error + quality, rangefinder, baro drift, magnetometer bias, depth fan |
| Failure injection | `gdnav/sim/sim_vehicle.py` (`FAULTS`) | camera covered / dropout, dim lights, link loss, featureless floor, wind gust, low battery, false landmarks |

## Conventions

- World frame is **local NED** in metres from the takeoff point (x north, y east, z down), and body
  frame is FRD, both as in PX4. Altitude above ground is `-z`.
- The flight controller and the mission logic only ever see the **estimate**. Ground truth is used
  only for metrics, the dashboard and the "dead-reckoning only" comparison estimator.
- All timing uses `vehicle.now()`, which is simulation time in the simulator and a monotonic clock
  with PX4.
