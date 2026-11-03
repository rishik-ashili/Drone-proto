import math

import numpy as np
import pytest

from gdnav.common.config import MODELS_DIR, load_sim_config
from gdnav.interface.vehicle import CameraFrame
from gdnav.perception.color_detector import ColorDetector
from gdnav.perception.cv_node import CvNode
from gdnav.sim.camera import DownwardCamera
from gdnav.sim.world import World


@pytest.fixture(scope="module")
def setup():
    cfg = load_sim_config()
    world = World.load("indoor_arena")
    return cfg, world, DownwardCamera(world)


def _observe(node, cam, pos, att):
    img = cam.render(np.array(pos), np.array(att), noise=3)
    rng = -pos[2] / (math.cos(att[0]) * math.cos(att[1]))
    res = None
    for _ in range(4):  # temporal filter needs consecutive frames
        res = node.process(CameraFrame(0.0, img, np.array(att), rng), np.array(pos))
    return res


def test_aruco_fix_accuracy(setup):
    cfg, world, cam = setup
    node = CvNode(cfg["perception"], world, cfg["sensors"]["camera"])
    pos, att = [9.1, 8.7, -2.5], [0.04, -0.03, 0.5]
    res = _observe(node, cam, pos, att)
    tag = next(o for o in res.observations if o.name == "tag_07")
    assert tag.confirmed
    assert np.linalg.norm(tag.fix_xy - np.array(pos[:2])) < 0.05
    assert abs(math.degrees(tag.yaw_fix - att[2])) < 1.0


@pytest.mark.parametrize("name,xy", [("pad_red", (9.0, 0.5)), ("pad_blue", (1.0, 9.5)), ("cross_c", (5.0, 4.5))])
@pytest.mark.parametrize("yaw", [0.0, 0.8, 2.5])
def test_colour_fallback_detects_pads_any_rotation(setup, name, xy, yaw):
    _, world, cam = setup
    img = cam.render(np.array([xy[0] + 0.2, xy[1] - 0.1, -2.5]), np.array([0.02, -0.02, yaw]))
    cls = world.landmark(name).cls
    assert cls in [d.cls for d in ColorDetector().detect(img)]


@pytest.mark.skipif(not (MODELS_DIR / "landmarks_yolov8n.onnx").exists(), reason="landmark model not trained")
@pytest.mark.parametrize("name", ["home_pad", "pad_red", "pad_blue", "cross_c", "tag_03"])
def test_yolo_detects_every_landmark_class(setup, name):
    cfg, world, cam = setup
    from gdnav.perception.yolo_detector import YoloDetector

    det = YoloDetector(MODELS_DIR / "landmarks_yolov8n.onnx")
    lm = world.landmark(name)
    img = cam.render(np.array([lm.x + 0.3, lm.y - 0.2, -2.5]), np.array([0.03, 0.02, 1.0]))
    assert lm.cls in [d.cls for d in det.detect(img)]


def test_floor_map_matching_recovers_drift(setup):
    from gdnav.perception.map_matcher import FloorMap, MapMatcher

    _, world, cam = setup
    mm = MapMatcher(cam.K)
    fmap = FloorMap(world.x_min, world.x_max, world.y_min, world.y_max, 0.01)
    for x in np.arange(2.0, 7.1, 1.0):  # teach flight over a patch of floor
        for y in np.arange(-1.0, 4.1, 1.0):
            pos, att = np.array([x, y, -2.5]), np.array([0.02, -0.01, 0.3])
            mm.teach(fmap, cam.render(pos, att, light=0.9, noise=4), pos, att)
    true_pos, att = np.array([4.6, 1.3, -2.4]), np.array([0.03, 0.02, 1.2])
    img = cam.render(true_pos, att, light=1.1, noise=3)
    est = true_pos + np.array([0.5, -0.4, 0.0])           # 0.64 m of drift
    est_att = att + np.array([0.0, 0.0, math.radians(3)])
    fix = mm.match(fmap, img, est, est_att, float(est_att[2]))
    assert fix is not None
    assert np.linalg.norm(fix.xy - true_pos[:2]) < 0.05
    assert abs(math.degrees(fix.yaw - att[2])) < 1.0


def test_charuco_calibration_recovers_intrinsics(tmp_path):
    """Render the printed ChArUco board through the camera model and calibrate it back."""
    import cv2
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibrate_camera as cc
    from gdnav.sim.world import World as W

    img = cc.board().generateImage((700, 980), marginSize=40)
    res = 0.035 * 5 / (700 - 80)
    tex = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rows, cols = tex.shape[:2]
    w = W(name="board", x_min=-rows * res / 2, x_max=rows * res / 2, y_min=-cols * res / 2, y_max=cols * res / 2,
          res=res, landmarks=[], obstacles=[])
    w.texture = tex
    cam = DownwardCamera(w, rng=np.random.default_rng(0))
    rng = np.random.default_rng(2)
    views = [cam.render(np.array([rng.uniform(-.08, .08), rng.uniform(-.08, .08), -rng.uniform(.35, .5)]),
                        np.array([rng.uniform(-.35, .35), rng.uniform(-.35, .35), rng.uniform(-.4, .4)]), noise=1)
             for _ in range(16)]
    c = cc.calibrate(views)
    assert abs(c["K"][0][0] - cam.K[0, 0]) / cam.K[0, 0] < 0.03
    assert c["rms_px"] < 1.0


def test_calibrated_distorted_camera_still_localizes(setup, tmp_path):
    import cv2
    import yaml

    cfg, world, cam = setup
    K = cam.K.copy()
    dist = np.array([-0.28, 0.08, 0.0, 0.0, 0.0])            # strong barrel distortion
    calib = tmp_path / "calib.yaml"
    calib.write_text(yaml.safe_dump({"width": 640, "height": 480, "K": K.tolist(), "dist": dist.tolist()}))
    pos, att = np.array([9.2, 8.6, -2.5]), np.array([0.03, -0.02, 0.4])
    ideal = cam.render(pos, att, noise=2)
    # make the distorted image the real lens would produce
    uu, vv = np.meshgrid(np.arange(640, dtype=np.float32), np.arange(480, dtype=np.float32))
    pts = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1)[:, None, :], K, dist, P=K).reshape(480, 640, 2)
    distorted = cv2.remap(ideal, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR)
    rng_m = -pos[2] / (math.cos(att[0]) * math.cos(att[1]))
    results = {}
    for name, c in (("raw", None), ("calibrated", str(calib))):
        node = CvNode(cfg["perception"], world, dict(cfg["sensors"]["camera"], calibration=c))
        for _ in range(4):
            r = node.process(CameraFrame(0.0, distorted, att, rng_m), pos)
        tag = [o for o in r.observations if o.name == "tag_07"]
        results[name] = np.linalg.norm(tag[0].fix_xy - pos[:2]) if tag else 9.9
    assert results["calibrated"] < 0.05
    assert results["calibrated"] < results["raw"]
