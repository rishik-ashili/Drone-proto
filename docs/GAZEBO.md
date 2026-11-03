# PX4 SITL + Gazebo: what to run on your machine

The built-in simulator runs our own model of PX4. This guide runs the **real PX4 firmware
(v1.17.0)** in software-in-the-loop, so the mission code talks to the same autopilot that will fly
on the Pixhawk. It needs Ubuntu 22.04 or 24.04 (native, or a VM with 3D acceleration), about 15 GB of
disk and roughly an hour the first time.

> These steps could not be run in the cloud environment that built this repository (the Gazebo
> package server is blocked there and building PX4 needed a permission that was not granted). The
> files are written against the PX4 v1.17.0 source, but expect to fix small things on the first run.
> Send me the terminal output of any step that fails.

## 0. One-time setup (about 30–45 min)

```bash
cd ~/Drone-proto
./scripts/setup.sh --px4                 # our Python environment + MAVSDK
source .venv/bin/activate
./scripts/px4_sitl.sh setup              # clones PX4 v1.17.0 to ~/PX4-Autopilot, installs the
                                         # toolchain + Gazebo Harmonic, builds px4_sitl
sudo apt install python3-gz-transport13 python3-gz-msgs10   # Gazebo camera into Python
```

Log out and back in once (the PX4 setup adds you to the `dialout` group). Optionally install
[QGroundControl](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html);
it connects automatically to SITL on UDP 14550.

To use `mavsdk_grpc` inside our virtual environment with the system Gazebo bindings, create the
venv with `python3 -m venv --system-site-packages .venv`, or run `run_px4.py` with the system
`python3` after `pip install --user -r requirements-px4.txt`.

## Step 1: fastest check, no Gazebo (SIH): about 5 min

PX4's built-in physics. GPS is simulated and on, so this checks only the **Offboard interface**:
arming, setpoint streaming, the state machine and landing. This is the Phase 0 "hello waypoint" on
real PX4 firmware.

```bash
# terminal 1
./scripts/px4_sitl.sh sih
# terminal 2
source .venv/bin/activate
python scripts/run_px4.py --mission hello_square
```

**Expected:** "connected via udp://:14540", the state machine runs PREFLIGHT → TAKEOFF →
GOTO_WAYPOINT ×4 → LAND, then a summary is printed. A log is written to `logs/px4_hello_square_*.jsonl`
(`python scripts/make_report.py --latest`). The drone is visible in QGroundControl.

## Step 2: GPS-free flight in Gazebo (stock x500 with optical flow): about 10 min

```bash
# terminal 1
./scripts/px4_sitl.sh run                # make px4_sitl gz_x500_flow (GPS disabled by the airframe)
# terminal 2
python scripts/apply_px4_params.py       # config/px4/gps_denied_flow.params
# in terminal 1 (pxh> prompt):  reboot        (or Ctrl+C and start again)
python scripts/run_px4.py --mission hello_square
```

In the `pxh>` console, before flying, check that the estimator runs without GPS:

```
pxh> listener estimator_status_flags     # cs_opt_flow: True, cs_gps: False
pxh> listener vehicle_local_position     # xy_valid: True
pxh> listener sensor_optical_flow
```

**Exit criterion P0:** fly `hello_square` 5 times in a row, with reach error under 0.5 m in the
summary. Gazebo is the ground truth here: check the drone's pose in the Gazebo GUI entity tree.

## Step 3: our arena, landmarks and downward camera (P1 closed loop): about 10 min

```bash
# terminal 1: exports worlds/indoor_arena.yaml to Gazebo, installs our model, starts PX4 + Gazebo
./scripts/px4_sitl.sh gdnav indoor_arena
# terminal 2
python scripts/apply_px4_params.py       # then 'reboot' in pxh>, once
python scripts/run_px4.py --mission investor_demo --gz-camera
```

What this sets up:

| File | Made by | Contents |
|---|---|---|
| `Tools/simulation/gz/worlds/gdnav_indoor_arena.sdf` | `scripts/export_gazebo.py` | 20 × 20 m floor textured with the same image the built-in simulator renders (all landmarks), 3 pillars, light |
| `Tools/simulation/gz/models/x500_flow_cam_down/` | same | PX4 `x500_flow` (optical flow + LW20 rangefinder) plus a downward 640×480, 80° camera on `/gdnav/down_camera/image` |
| airframe | PX4 | `4021_gz_x500_flow`, launched with `PX4_SIM_MODEL=gz_x500_flow_cam_down` |

Check, in a third terminal:

```bash
gz topic -l | grep down_camera                       # /gdnav/down_camera/image
gz topic -e -t /gdnav/down_camera/image -n 1 | head  # width: 640 height: 480
```

**Check the floor orientation once.** Gazebo is ENU (x east, y north), our missions are NED
(x north, y east). After takeoff the drone should be over the helipad. `pad_red` must be **9 m
north**, which is Gazebo +y and the direction QGroundControl shows as north. If the floor is
mirrored or rotated, the texture mapping differs in your Gazebo version: tell me what you see,
because it's a one-line fix in `export_gazebo.py` (`floor_yaw`, or flipping the PNG).

**Exit criterion P1:** the mission runs end to end with landmark detection driving the state
machine (ALIGN after each waypoint, and landmark fixes accepted in the summary).

## Headless (no GUI, for a server or a weak laptop)

```bash
./scripts/px4_sitl.sh headless gdnav indoor_arena
```

## Things that differ from the built-in simulator

- The camera image is rendered by Gazebo (lighting, shadows), so it is a good test of the YOLO
  model on images it was not trained on. If detection is weak, record frames with
  `scripts/record_dataset.py --gz-camera` and add them to the dataset (docs/ML.md).
- Landmark fixes go to EKF2 as `VISION_POSITION_ESTIMATE`. Check that they are fused:
  `listener estimator_aid_src_ev_pos` (innovation and `fused: True`).
- Ground truth is not read by `run_px4.py`, so reach and RTS errors are estimate-based. Compare
  with the Gazebo pose.
- Collision prevention (`CP_DIST`) needs a distance sensor pointing forward. The stock `x500_flow`
  has only the downward LW20, so pilot flying in Gazebo is not protected from the pillars.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Gazebo gz sim not found` | `./scripts/px4_sitl.sh setup` again, or follow gazebosim.org/docs/harmonic/install_ubuntu |
| Arming denied: `No valid local position` | Parameters not applied or not rebooted; check `listener estimator_status_flags` |
| `Offboard rejected` | `run_px4.py` streams setpoints first; check that nothing else (QGC joystick) is sending |
| `could not subscribe to /gdnav/down_camera/image` | Not running `gdnav` mode, or the bindings are missing: `sudo apt install python3-gz-transport13 python3-gz-msgs10` |
| Black or grey floor | Albedo path. `install_gazebo_assets.sh` writes an absolute `file://` path: check that the PNG exists |
| Everything very slow | Use `headless`. Gazebo needs a GPU, and a VM without 3D acceleration renders on the CPU |
