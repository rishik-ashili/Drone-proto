# Flight checklists

Print these. Tick every line. One person reads, the pilot confirms. Abort on any "no".
Plan section 12: props off for bench work, tether for first hovers and every new software build,
and a trained pilot with RC override and a kill switch on every flight.

## A. Before a test day

- [ ] Test card(s) for today selected (docs/TEST_CARDS.md), with pass criteria written down
- [ ] Software version noted: `git rev-parse --short HEAD` = ________; working tree clean
- [ ] Simulator rehearsal of today's mission passed:
      `python scripts/run_sim.py --mission <m>` and `python scripts/validate_mission.py <m>`
- [ ] PX4 parameters match the repo: `python scripts/bench_check.py --url <url>` (all PASS)
- [ ] Landmarks printed at real size (`scripts/make_printables.py`), scale bar measured = 10.0 cm
- [ ] Batteries charged (flight + RC + companion), storage voltage noted
- [ ] Venue: permission, floor texture OK for optical flow, lighting even, no reflections
- [ ] Outdoor only: Digital Sky zone checked (green), Drone Rules 2021 compliance, NOTAM, weather

## B. Arena setup

- [ ] Floor grid / reference points measured (tape measure from the takeoff mark)
- [ ] Each landmark placed at its mission coordinates (NED from takeoff: x north, y east) and
      orientation (yaw from the world file), taped flat
- [ ] Obstacles in place and in the world file; safety net or perimeter clear
- [ ] Takeoff mark = helipad centre; drone nose pointing north (yaw 0) at the start
- [ ] Spectators behind the safety line, fire extinguisher and first-aid kit present

## C. Bench (props OFF)

- [ ] **Props removed** (show them to the safety officer)
- [ ] Companion powered from its own regulator; Pixhawk on battery or USB
- [ ] `bench_check.py --vision-test` all PASS; `--offboard-test` motors respond, disarm on stop
- [ ] Camera: `python scripts/benchmark_perception.py` FPS ≥ camera rate; image sharp, not upside down
- [ ] Move the frame forward by hand: local position x increases; rotate clockwise: yaw increases
- [ ] Failsafes: unplug the companion link → PX4 enters Hold (COM_OBL_RC_ACT); RC kill switch cuts motors

## D. Pre-flight (props ON)

- [ ] Props correct direction, tight, undamaged; frame, arms and wiring secure; battery strapped
- [ ] Tether attached (first hovers and new builds)
- [ ] RC transmitter on, mode switch = Position, kill switch tested **before** arming
- [ ] QGroundControl: no pre-arm errors; "Local position" valid **with GPS disabled**
- [ ] Companion service in STANDBY (`curl http://<companion>:8081/status`, state = IDLE)
- [ ] Logging on: PX4 SD card logging + companion `logs/`
- [ ] Everyone clear, call "ARMING"

## E. In flight

- [ ] Pilot's hand on the mode switch the whole time. Take over (Position mode) on any surprise:
      unexpected heading, oscillation, drift towards people or obstacles, failsafe text
- [ ] Observer watches the drone; a second person calls out the companion state / faults
- [ ] Battery: land at 30 % (RTS triggers at 25 %, land at 15 %)

## F. Post-flight

- [ ] Disarmed, battery disconnected, props safe
- [ ] Note in the test log: time, mission, result, anything unusual
- [ ] Download PX4 log: `python scripts/analyze_ulog.py <log>.ulg`
- [ ] Companion report: `python scripts/make_report.py logs/<file>.jsonl`
- [ ] Measure the landing offset from the takeoff mark with a tape (return-to-start error)
- [ ] Record new camera frames of failures for the dataset (`scripts/record_dataset.py`)
