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
