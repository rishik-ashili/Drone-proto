# What you need to run and do

Everything that can be done without hardware is in the repository. These are the steps only you can
do, in order. Send me the terminal output of anything that fails.

## Now (laptop only)

1. **Run the demo:** `./scripts/setup.sh && source .venv/bin/activate && python scripts/run_dashboard.py --open`
   - Try *Nav: Stereo VIO*, *Floor map*, *Landmark AI* off, **Take control** (W/A/S/D, R/F, Q/E or
     a gamepad), clicking the map, and the failure buttons.
2. **Run the test campaign once on your laptop:** `python scripts/benchmark.py` (15–25 min). This
   regenerates `docs/benchmark_results.md` with your machine's numbers.
3. **PX4 SITL** (Ubuntu 22.04/24.04), following [GAZEBO.md](GAZEBO.md):
   - step 1 (SIH, no Gazebo): `./scripts/px4_sitl.sh sih` + `python scripts/run_px4.py --mission hello_square`
   - step 2 (Gazebo, optical flow, no GPS): closes the **P0 exit criterion on real PX4 firmware**
   - step 3 (our arena + downward camera): closes **P1 in SITL**
4. **Datasets:** from your machine (Zenodo is reachable there), optionally
   `python scripts/fetch_public_datasets.py kios_landing_pad`. Check its licence and only map it to
   a class you will actually fly.
5. **Tag the pre-hardware release** (my session could push only the working branch, not tags):
   `git tag -a v0.3.0-sim 953d17e -m "Pre-hardware release" && git push origin v0.3.0-sim`,
   or create the release on GitHub from commit `953d17e`.
6. **Decide the open items from the plan (section 13):**
   - Which features the reference video shows.
   - Camera, flow sensor, rangefinder and companion board models.
   - Landmark set and waypoint count.
   - Your deadline.

## When the parts arrive (P1–P2)

| Step | Command / action |
|---|---|
| Flash PX4 v1.17 to the Pixhawk 6C; calibrate in QGC | QGroundControl |
| Apply our parameters | `python scripts/apply_px4_params.py --url serial:///dev/ttyACM0:921600`, then reboot |
| Bench check (props **off**) | `python scripts/bench_check.py --url serial:///dev/ttyACM0:921600 --vision-test --offboard-test` |
| Print landmarks + calibration board | `python scripts/make_printables.py`; `python scripts/calibrate_camera.py --make-board` |
| Calibrate the downward camera | `--capture 0`, then `--images datasets/calib`; set `sensors.camera.calibration` |
| Speed on the companion board | `python scripts/benchmark_perception.py` (record for T1.3) |
| Record real frames at flight height | `python scripts/record_dataset.py --camera 0 --fps 3` (several days, floors, lights) |
| Correct the pre-labels | CVAT / Label Studio / Roboflow (YOLO 1.1 import/export) |
| Retrain on synthetic + real | `python scripts/merge_datasets.py --add datasets/landmarks --add datasets/raw/<dir>:auto` then `python scripts/train_landmark_model.py --data datasets/merged/data.yaml` |
| Install on the companion as a boot service | `sudo ./scripts/deploy_companion.sh` (starts in STANDBY: never flies on its own) |
| Tethered hover, then flights | docs/CHECKLISTS.md + docs/TEST_CARDS.md (T1.4, T2.2, T2.3) |
| After each flight | `python scripts/analyze_ulog.py <log>.ulg` and `python scripts/make_report.py --latest` |

## Jetson phase (P3)

Follow [jetson/README.md](../jetson/README.md): JetPack 6, Kalibr camera–IMU calibration, Isaac ROS
Visual SLAM, `docker/Dockerfile.jetson` (or a native colcon build), then
`ros2 launch gdnav_ros gdnav.launch.py nav_source:=vio`, and `scripts/trt_engine_builder.py --fp16`.

## Things I could not do from here (and why)

| Item | Reason |
|---|---|
| Run PX4 SITL / Gazebo | Gazebo's package server is blocked from my environment, and building PX4 needed a permission that wasn't granted |
| Download Zenodo / Roboflow / Kaggle / Hugging Face datasets | Those hosts are blocked from my environment (only GitHub was reachable) |
| Real photos of your landmarks | Needs your camera at flight height |
| Anything on a Pixhawk, Pi or Jetson | Hardware |
| Measure real sensor noise for `config/sim.yaml` | Needs bench logs (`analyze_ulog.py` will show them) |
