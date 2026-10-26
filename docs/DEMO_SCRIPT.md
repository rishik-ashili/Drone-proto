# Investor demo script (about 6 minutes)

**Setup before the meeting.** Run `python scripts/run_dashboard.py --open` on the laptop and connect
a second screen if you have one. As a backup, record a video with
`python scripts/run_sim.py --record demo.mp4` and keep it on the desktop.

---

### 1. The problem (30 s, mission idle)

> "Drones navigate with GPS. Indoors, under bridges, in warehouses, in mines, or anywhere GPS is
> jammed, they are blind. We are building navigation that does not use GPS at all."

Point at the 3D view: a 20 × 20 m indoor arena, four landmarks, three pillars, and a home helipad.

### 2. Start the mission (2 min)

Press **▶ Start mission**, with speed at 1× or 2×.

- **Takeoff.** "Position comes only from an optical-flow camera, a rangefinder and the IMU, as on our
  real Pixhawk. Watch the *PX4 mode*: Offboard, meaning our computer is flying the drone."
- **First leg.** "Every sensor has small errors, so the position estimate slowly drifts. The orange
  line in the accuracy chart is where the drone *thinks* it is without our AI, already tens of
  centimetres off."
- **Pillar.** The planner pill turns amber and the red depth rays appear. "It sees the pillar with
  its depth sensor and flies around it."
- **Landmark found.** Look at the camera panel. "Our neural network detects the landmark and
  measures its offset in centimetres. That fix goes into the flight controller's estimator, and the
  blue error line drops back to a few centimetres."
- **ALIGN.** "The drone servos precisely over the landmark. Waypoint reach error: about 5–10 cm."

### 3. Break things on purpose (2 min)

Use the *Failure injection* panel during an ALIGN phase:

1. **Camera covered.** The state machine goes to FAILSAFE HOLD within 0.05 s. Clear it: the drone
   resumes. "Every failure has a defined, tested response."
2. **Companion ↔ Pixhawk link lost.** PX4's own failsafe holds position, then our software takes
   back control when the link returns.
3. **Inject false-positive detections.** The red boxes are fake detections. "The estimator's
   statistical gate rejects them. A wrong detection can't drag the drone away."
4. **Battery drops to 22 %.** The drone abandons the mission and returns home.

### 4. Results (1 min)

When it lands: *Return-to-start error* (typically 5–20 cm after about 45 m) and *Waypoint reach
error*. Then untick **Vision correction** (this resets the run) and fly again to show the drone
missing landmarks and landing off the pad.

> "Same drone, same sensors. The only difference is our vision AI. That's the product."

### 5. Credibility (30 s)

- "This is the code that runs on the drone. It also drives the real PX4 autopilot (SITL and Pixhawk)
  through the same interface."
- "Our automated test campaign checks every phase exit criterion and every failsafe": show
  `docs/benchmark_results.md`.
- Roadmap: Jetson Orin with stereo visual-inertial odometry (Phase 3), then obstacle avoidance and
  drift correction at scale (Phase 4).

---

**Talking points about the AI:**

- The landmark detector is a YOLOv8-nano network, fine-tuned from open-source COCO weights on
  synthetic images that our simulator renders and labels automatically. Thousands of labelled images
  cost nothing to make.
- It runs in about 10 ms per frame on a laptop CPU, and will run under 5 ms on the Jetson with
  TensorRT.
- AprilTag/ArUco markers provide a second, non-ML source of truth for cross-checking.
