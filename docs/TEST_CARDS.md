# Test cards

One card per test. Progression for every new capability (plan section 10): simulation →
replay → bench (props off) → tethered → free flight. Copy a card into the test log, fill in the results.

| ID | Phase | Test | Pass criterion (from the plan) | Tool |
|---|---|---|---|---|
| T0.1 | P0 | Square ×5 in the built-in simulator | reach error < 0.5 m every waypoint, 5 in a row | `scripts/benchmark.py` |
| T0.2 | P0 | Square on PX4 SITL (SIH, then Gazebo `gz_x500_flow`) | same, logs saved and replayable | `docs/GAZEBO.md` steps 1–2 |
| T1.1 | P1 | Landmark mission on PX4 SITL + Gazebo camera | detection drives the state machine end to end | `px4_sitl.sh gdnav`, `run_px4.py --gz-camera` |
| T1.2 | P1 | Bench: setpoints on the real Pixhawk | Offboard engages, setpoints echo, disarm on stop | `bench_check.py --offboard-test` |
| T1.3 | P1 | Detection rate on the POC companion | measured FPS ≥ camera rate, recorded | `benchmark_perception.py` |
| T1.4 | P1 | Tethered hover, GNSS disabled | stable hover 60 s, eph < 0.3 m | QGC + `analyze_ulog.py` |
| T2.1 | P2 | Model v1 on real images | precision / recall / mAP on a held-out real set | `train_landmark_model.py` (val on real) |
| T2.2 | P2 | 3 consecutive arena flights, N waypoints, no GNSS | all waypoints, RTS error logged each flight | `run_px4.py --mission ...` |
| T2.3 | P2 | Each failsafe, tethered | expected response (table below), time measured | sim matrix first: `benchmark.py` |
| T3.1 | P3 | VIO hold vs. flow hold | VIO drift (m/min) lower than flow | `missions/hover_test.yaml` |
| T3.2 | P3 | Detection + VIO concurrently on the Orin | ≥ 30 FPS detection, ≥ 30 Hz odometry into PX4 | `benchmark_perception.py`, `ros2 topic hz` |
| T3.3 | P3 | Drift over the standard route | drift as % of distance, recorded | `analyze_ulog.py`, tape at landing |
| T4.1 | P4 | Failure injection after the software freeze | all failsafes verified again | below |
| T4.2 | P4 | Demo mission repeated | metrics within the agreed targets, N times | `docs/DEMO_SCRIPT.md` |

## Failsafe expectations (T2.3, T4.1)

| Fault | How to inject (real) | Sim | Expected response |
|---|---|---|---|
| Camera covered | cap over the lens during ALIGN | `camera_covered` | FAILSAFE_HOLD within 0.1 s; resume after 1 s clear; LAND if > 10 s |
| Camera frames stop | unplug camera / kill driver | `camera_dropout` | FAILSAFE_HOLD within 1 s |
| Companion link lost | unplug TELEM2 | `link_loss` | PX4 Hold after COM_OF_LOSS_T (1 s); companion re-engages when back |
| Low battery | PX4 `BAT_LOW_THR`, or a half-charged pack | `low_battery` | return to start at 25 %, land at 15 % |
| Optical flow lost | cover the flow sensor | `featureless_floor` | landmarks / floor map keep the estimate; hold if eph > 0.9 m |
| False detections | printed decoy (a wrong-colour pad) | `false_landmark` | EKF gate rejects the fix; the mission is unaffected |
| VIO tracking lost | cover the stereo camera | `vio_tracking_loss` | flow fallback, VIO re-anchored on recovery, error < 0.5 m |
| Dim light | switch off half the lights | `dim_lights` | mission completes, or holds if detection is lost |
| Wind gust | fan at 3 m/s | `wind_gust` | mission completes, no contact |

---

### Card template

```
ID / date / location:            Software (git) / PX4 version:
Objective:
Setup (airframe, batteries, tether?, landmarks, lighting):
Steps:
  1.
Pass criterion:
Result (numbers):                 PASS / FAIL
Logs: PX4 ________.ulg   companion logs/________.jsonl
Notes / anomalies / follow-ups:
```
