# GPS-denied waypoint navigation: technical report (pre-hardware, v0.3)

## 1. Summary

A quadcopter flies a list of waypoints, given in metres from its takeoff point, without GNSS. PX4
estimates position from optical flow, a rangefinder and the IMU. A companion computer detects
known landmarks with a YOLOv8-nano network and ArUco markers and converts each detection into an
absolute position fix. That fix removes the drift dead reckoning accumulates.

Everything up to the hardware boundary is implemented and tested in a software-in-the-loop digital
twin. That includes the Phase 3 (stereo VIO) and Phase 4 (map-based drift correction) capabilities,
as behavioural models. The same companion code runs against real PX4 through MAVSDK (SITL or a
Pixhawk on a Pi) or ROS 2 + uXRCE-DDS (Jetson). None of it has flown yet.

Headline simulator results (`docs/benchmark_results.md`: mean of 3 seeds on the investor_demo route,
about 44 m, 4 landmarks, 3 pillars):

| Configuration | Mean waypoint reach error | Return-to-start error | Max estimate error |
|---|---|---|---|
| Optical flow + IMU only (dead reckoning) | 0.57 m | 0.19 m | 0.86 m |
| + landmark AI (YOLO + ArUco) | 0.09 m | 0.05 m | 0.53 m |
| + floor-map matching, no landmarks | 0.07 m | 0.16 m | 0.17 m |
| Stereo VIO model only | 0.07 m | 0.09 m | 0.18 m |
| Stereo VIO model + landmark AI | 0.08 m | 0.06 m | 0.13 m |

Hover drift over 60 s: 0.12 m/min on optical flow, 0.02 m/min with the floor map, and 0.01 m/min with
the VIO model.

Every failsafe in the matrix produces its specified response. Landmark detector on a held-out
synthetic split: precision 0.97, recall 0.98, mAP50 0.99, mAP50-95 0.89. Perception runs in
17–28 ms per frame on a 4-core laptop CPU.

## 2. Architecture

The layers follow the plan (section 3), with one vehicle interface separating the companion from
the autopilot:

```
camera ─► cv_node (YOLO + ArUco, temporal filter) ─► landmark_localizer ─► map association
   │                                                        │ fix (x, y, heading, sigma)
   │      floor-map matcher (teach and repeat) ─────────────┤
   │      VIO odometry ─► landmark/VIO fusion ◄─────────────┘
   ▼                              │ external vision (EKF2_EV_CTRL)
health_monitor ─► mission state machine ─► Offboard setpoints ─► PX4 (EKF2, control, failsafes)
```

`VehicleInterface` (`gdnav/interface/vehicle.py`) has three implementations: `SimVehicle` (built-in
physics, EKF2 and controller analogues), `Px4MavsdkVehicle` (MAVLink: setpoints, `VISION_POSITION_ESTIMATE`,
`ODOMETRY`, manual control) and `Px4RosVehicle` (px4_msgs over uXRCE-DDS for PX4 v1.17, including
the `_v1` topic suffixes). Timing uses `vehicle.now()` everywhere.

## 3. State estimation without GNSS

The estimator (`gdnav/sim/estimator.py`) mirrors EKF2's sensor set:

- **Prediction:** IMU specific force rotated by the attitude, with body-frame accelerometer bias states.
- **Updates:** optical-flow velocity scaled by the rangefinder, rangefinder and barometer height,
  magnetometer heading, and external vision (position, heading, velocity).
- **Gating:** every external update passes a χ² innovation gate (`EKF2_EVP_GATE` analogue), so
  false detections cannot move the estimate. The EKF2 reset path (odometry `reset_counter`) handles
  legitimate jumps.
- **Sensor errors modelled:** flow scale error and noise that grows with height, flow quality loss,
  accelerometer bias random walk, gyro bias, a 3.5° magnetometer bias, and barometer drift.

The magnetometer bias produces the dominant dead-reckoning error: a heading error turns forward
motion into sideways drift.

## 4. Perception

- **Detectors.** YOLOv8n at 320 px through ONNX Runtime (or OpenCV DNN, or the TensorRT
  execution provider), plus ArUco (DICT_4X4_50) with sub-pixel corners. An HSV colour fallback
  covers boards with no model.
- **Localization.** The pixel is back-projected through calibrated intrinsics, rotated by the
  estimated attitude, and intersected with the ground at the rangefinder height. The fix is the
  landmark's map position minus that offset. For ArUco tags, heading also comes from the marker's
  edge direction. Accuracy on rendered frames: under 1 cm in position and under 0.2° in heading.
- **Robustness.**
  - Temporal filter: three consecutive consistent frames are needed before a detection counts.
  - Association gate: a fix is accepted only if a mapped landmark of that class is within 2.5 m.
  - EKF innovation gate: rejects fixes that disagree too much with the current estimate.
- **Latency compensation.** A fix is applied as a correction of the estimate at exposure time,
  which is equivalent to EKF2's delayed fusion with `EKF2_EV_DELAY`.
- **Lens calibration.** ChArUco calibration (`scripts/calibrate_camera.py`) feeds a precomputed
  undistortion remap. A test with strong barrel distortion confirms fixes stay under 5 cm.

## 5. Drift correction beyond landmarks

- **Stereo VIO (Phase 3).** A behavioural model of cuVSLAM output: scale error, distance-scaled
  random walk, heading drift, and tracking loss with frame re-initialisation. EKF2 accepts one
  external-vision stream, so landmark fixes update a 3-DoF correction of the VIO frame (a small
  Kalman filter whose process noise grows with distance flown). The corrected odometry is what
  PX4 receives. On re-initialisation the frame is re-anchored to the flight controller's current
  estimate, which carried on with optical flow during the outage. The same module feeds the
  ROS 2 node.
- **Teach-and-repeat floor-map matching (Phase 4).**
  - Frames are orthorectified at the estimated pose (CLAHE contrast enhancement, 1 cm/px).
  - **Teach:** while a landmark fix is recent, frames are painted into a georeferenced mosaic.
  - **Repeat:** each frame is matched against the mosaic (ORB features, RANSAC similarity). The
    offset gives a position and heading fix.
  - Offline tests: 0.8 m of drift corrected to about 3 mm, and heading to under 0.3°.
  - Limitation (measured, then fixed): the map is only as accurate as the teach flight. A map
    taught on optical flow degraded VIO, so map fixes now carry the map's own uncertainty, and
    the shipped map was taught with VIO + landmarks.
  - The same matcher accepts any orthophoto or satellite tile (the outdoor variant in the plan).
    AnyLoc/DINOv2 retrieval would add appearance invariance and is not implemented.

## 6. Navigation and control

- **Mission states:** TAKEOFF, GOTO_WAYPOINT, SEARCH_LANDMARK (expanding square), ALIGN (PI visual
  servoing on the metric offset, then hold), NEXT_WAYPOINT, RETURN_TO_START (Offboard in the local
  frame, because PX4's Return needs a global position), PRECISION_LAND, LAND.
- **Operator states:** MANUAL (pilot Position mode, where companion failsafes defer to the pilot),
  USER_GOTO and USER_HOLD (click on the map).
- **Obstacles.** A reactive repulsive/tangential planner on a forward depth fan, standing in for
  EGO-Planner with the same velocity interface. In manual flight, PX4-style collision prevention
  (`CP_DIST`) limits the pilot's velocity toward obstacles.
- **Geofence.** A software geofence clamps setpoints. A breach triggers return-to-start.

## 7. Safety

The health monitor watches:

- telemetry age (link)
- battery (return to start at 25 %, land at 15 %)
- camera frames and image statistics (lens covered)
- estimator position uncertainty
- the geofence
- VIO tracking
- PX4 failsafe flags

The chain is fault → FAILSAFE_HOLD → LAND if it persists for 10 s, with 1 s of hysteresis before
resuming. PX4's own failsafes are configured as backups: offboard loss → Hold, critical battery →
Land, geofence → Land. On a companion computer, the service starts in standby and never takes off
without an explicit command.

## 8. Machine learning

| Item | Detail |
|---|---|
| Classes | helipad, pad_red, pad_blue, cross_yellow, aruco_tag |
| Base model | Ultralytics YOLOv8n COCO-pretrained, 3.1 M parameters |
| Training data | 1,600 synthetic images rendered by the simulator with randomised floors, light, height, tilt, blur, JPEG and distractors |
| v1 result | 30 epochs on CPU. Held-out synthetic split: P 0.973, R 0.984, mAP50 0.994, mAP50-95 0.893 |
| External check | 260 MIT-licensed helipad images in a different visual style: v1 scores mAP50 0.995 on their test split |
| v2 | Fine-tuned on synthetic + external (see `models/v2`, below) |

The licence of Ultralytics (AGPL-3.0) matters for a proprietary product (docs/ML.md). Real images
of the printed landmarks are the most important missing data.

## 9. Verification status

| Level | State |
|---|---|
| Unit and integration tests | pytest suite: estimator, localizer, frames, fusion, map matching, calibration, ROS conversions, tools, end-to-end missions, failsafes, operator control |
| Simulator campaign | `scripts/benchmark.py` → `docs/benchmark_results.md` (P0, P1/P2, sources comparison, hold drift, failsafes) |
| PX4 SITL | Scripts and Gazebo assets written against v1.17.0; **not run** (blocked in the build environment) |
| ROS 2 / Jetson | Stub-tested only |
| Hardware | None |

## 10. Known limitations

- The flight dynamics, EKF and controller are analogues of PX4, not PX4 itself. Gains and noise are
  representative, not measured. Bench logs (`analyze_ulog.py`) should replace them.
- Camera rendering is geometric. There are no shadows, specular highlights or rolling shutter;
  Gazebo and real images will be harder.
- The VIO numbers come from a model with assumed parameters. Its 0.01 m/min hover drift is an
  input assumption, not evidence that VIO is better.
- Teach-and-repeat assumes the floor looks the same as during teaching.
- The obstacle planner is reactive and can be trapped by concave obstacles. Collision prevention
  covers only the forward sensor field.

## 11. Next steps

Run PX4 SITL (`docs/GAZEBO.md`), then work through the hardware bring-up in `docs/YOUR_TODO.md`:
the bench check, calibration, real data, tethered hover and the arena flights of test cards
T1.2–T2.3. Then measure the real sensor noise and feed it back into `config/sim.yaml`, so the
twin predicts the real drone.
