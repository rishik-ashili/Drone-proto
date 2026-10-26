import numpy as np

from gdnav.common.config import load_sim_config
from gdnav.sim.estimator import PositionEstimator


def test_external_vision_gate_rejects_outlier():
    cfg = load_sim_config()["estimator"]
    est = PositionEstimator(cfg)
    assert est.update_external_vision(0.0, np.array([0.03, -0.02]), 0.05)
    assert not est.update_external_vision(0.1, np.array([3.0, 3.0]), 0.05)  # false landmark
    assert np.linalg.norm(est.pos[:2]) < 0.1


def test_flow_bounds_velocity_uncertainty_at_rest():
    cfg = load_sim_config()["estimator"]
    est = PositionEstimator(cfg)
    for _ in range(2000):
        est.predict(np.array([0.0, 0.0, -9.81]), np.zeros(3), 0.0, 0.0, 0.005)
        est.update_flow(0.0, np.zeros(2), 0.02)
    assert est.Ph[2, 2] < 0.01
