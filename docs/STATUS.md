# Status against the phase plan

The plan has 48 deliverables and exit criteria. **21 are done, 20 are partly done, and 7 are left.**
All 7 that are left need physical hardware.

- **Done:** built and tested in the simulator.
- **Partly:** the code or files exist but have not run on real PX4, Gazebo or hardware; or the item
  passes only in simulation; or a different method was used than the plan named.
- **Left:** not possible without hardware.

## Week 1: P0 foundations (3 done · 2 partly · 0 left)

| Item | Status | Where / note |
|---|---|---|
| Repository skeleton, launch scripts | Done | `scripts/`, `px4_sitl.sh` (sih / run / gdnav) |
| Hello-waypoint mission | Done | `missions/hello_square.yaml` (sim); PX4 path `run_px4.py` |
| Common log format + replay | Done | `gdnav/common/logger.py`, `tools/replay.py`, `make_report.py` |
| Docker image | Partly | `docker/Dockerfile` written, not built here (no Docker daemon) |
| Exit: square ×5 < 0.5 m | Partly | Passes in the simulator; you run it on SITL: docs/GAZEBO.md step 2 |

## Weeks 2–3: P1 CV + waypoint POC (6 done · 2 partly · 3 left)

| Item | Status | Where / note |
|---|---|---|
| cv_node (YOLO + ArUco) | Done | `gdnav/perception/cv_node.py` |
| landmark_localizer | Done | + lens calibration and undistortion (`calibrate_camera.py`) |
| waypoint_manager + state machine | Done | + MANUAL, click-to-go, resume |
| Logger + replay | Done | |
| Pretrained YOLO nano + ArUco | Done | COCO yolov8n → custom model |
| Visual servoing on pixel offset | Done | ALIGN, PRECISION_LAND |
| Dataset v0 at flight height | Partly | 1,900 synthetic + 260 public (MIT) images; real frames via `record_dataset.py` |
| SITL with Gazebo camera into cv_node | Partly | World, model and `--gz-camera` bridge written; you run GAZEBO.md step 3 |
| Bench test on the real controller (props off) | Left | `bench_check.py --offboard-test` ready |
| Tethered hover, GNSS disabled | Left | Checklists + test card T1.4 ready |
| Detection rate measured on the POC companion | Left | `benchmark_perception.py` ready (laptop CPU: about 57 FPS) |

## Weeks 4–5: P2 own ML + firmware (6 done · 4 partly · 1 left)

| Item | Status | Where / note |
|---|---|---|
| Trained model v1 + evaluation report | Done | `models/landmarks_yolov8n.*` |
| PX4 parameters under version control | Done | `config/px4/gps_denied_flow.params` (checked against v1.17.0) |
| Health monitor + failsafe chain | Done | 10 injectable faults, automated matrix |
| Custom return-to-start (Offboard) | Done | + precision landing |
| Flight checklists, test cards | Done | `docs/CHECKLISTS.md`, `docs/TEST_CARDS.md` |
| Telemetry dashboard | Done | Plus pilot control |
| Real labelled dataset | Partly | Recorder with pre-labels, merge tool, public-set fetcher |
| Boot-time services on the companion | Partly | systemd units + `deploy_companion.sh` (standby: never flies on its own), untested on a device |
| Exit: RTS error each flight | Partly | Simulator only |
| Exit: each failsafe demonstrated | Partly | Simulator only (all pass) |
| Exit: 3 arena flights, no GNSS | Left | Hardware |

## Weeks 6–8: P3 Jetson Orin (1 done · 6 partly · 3 left)

| Item | Status | Where / note |
|---|---|---|
| Landmark correction fused with VIO | Done | `gdnav/bridge/landmark_vio_fusion.py`, tested in the simulator |
| Perception + mission on ROS 2 | Partly | `jetson/ros2/gdnav_ros` (stub-tested; px4_msgs fields checked against v1.17) |
| VIO → PX4 bridge, frames | Partly | Frame maths unit-tested; not run with ROS |
| EKF2 external-vision configuration | Partly | Parameters + docs; not flown |
| Benchmark POC vs Orin | Partly | `benchmark_perception.py`, `trt_engine_builder.py` |
| Docker, launch files, boot on the Jetson | Partly | `Dockerfile.jetson`, compose, launch, systemd, untested |
| Exit: VIO hold better than flow | Partly | Simulator VIO model says yes; it's a model until measured |
| cuVSLAM integration | Left | Needs the stereo camera + Jetson |
| Exit: ≥ 30 FPS detection + ≥ 30 Hz odometry | Left | Hardware |
| Exit: drift as % of distance | Left | Hardware (simulator numbers in benchmark_results) |

## Weeks 9–10: P4 precision + demo (5 done · 6 partly · 0 left)

| Item | Status | Where / note |
|---|---|---|
| AprilTag/ArUco drift anchors | Done | Position + heading fixes |
| Failure-injection tests | Done | Camera, frames, lights, link, flow, VIO, wind, battery, false detections |
| Scripted demo mission | Done | `investor_demo`, `docs/DEMO_SCRIPT.md` |
| Metrics report | Done | `docs/benchmark_results.md`, per-flight reports |
| Technical write-up | Done | `docs/TECHNICAL_REPORT.md` |
| Obstacle avoidance | Partly | Reactive planner + PX4-style collision prevention; not EGO-Planner |
| Teach-and-repeat drift correction | Partly | Floor-map matching (ORB + RANSAC) works; AnyLoc descriptors not used |
| Satellite-map matching | Partly | Same matcher accepts any orthophoto; no outdoor test |
| Final release tagged | Partly | `v0.3.0-sim` (pre-hardware); the final tag comes after the hardware freeze |
| Exit: demo repeated within targets | Partly | Simulator only |
| Exit: failsafes verified after freeze | Partly | Simulator only |

## Built beyond the plan

Manual Position-mode flying and click-to-go from the dashboard; PX4-style collision prevention;
standby companion service with a control API; printable landmarks; camera calibration; ULog
analysis; mission validator; Gazebo export of any world.
