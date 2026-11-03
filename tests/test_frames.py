import math

import numpy as np

from gdnav.bridge.frames import q_enu_flu_to_ned_frd, q_from_yaw, q_rotate, ros_odometry_to_px4, yaw_from_q


def test_position_enu_to_ned():
    pos, _, _ = ros_odometry_to_px4([1.0, 2.0, 3.0], q_from_yaw(0.0))
    assert np.allclose(pos, [2.0, 1.0, -3.0])


def test_heading_conversion():
    # ROS yaw 0 = facing east -> PX4 yaw +90 deg; ROS facing north -> PX4 yaw 0
    assert math.isclose(math.degrees(yaw_from_q(q_enu_flu_to_ned_frd(q_from_yaw(0.0)))), 90.0, abs_tol=1e-6)
    assert math.isclose(yaw_from_q(q_enu_flu_to_ned_frd(q_from_yaw(math.pi / 2))), 0.0, abs_tol=1e-6)


def test_moving_forward_increases_px4_x():
    # plan section 11: "check that moving forward increases PX4 X" (drone facing north)
    _, q, vel = ros_odometry_to_px4([0, 0, 0], q_from_yaw(math.pi / 2), [1.0, 0.0, 0.0])
    world_vel = q_rotate(q, vel)
    assert world_vel[0] > 0.99 and abs(world_vel[1]) < 1e-6
    assert np.allclose(q_rotate(q, [0, 0, 1]), [0, 0, 1])  # FRD down is NED down


def test_landmark_vio_fusion_removes_drift():
    from gdnav.bridge.landmark_vio_fusion import LandmarkVioFusion

    f = LandmarkVioFusion(init=(0.0, 0.0, 0.0))
    # the VIO frame drifts gradually to an offset of (0.5, -0.3) m and 2 deg over 30 m
    n = 100
    for i in range(n):
        k = i / (n - 1)
        drift, dyaw = k * np.array([0.5, -0.3]), k * math.radians(2.0)
        p_odom = np.array([i * 0.3, 0.1 * i, -2.0])
        f.on_odometry(p_odom, 0.0)
        c, s = math.cos(dyaw), math.sin(dyaw)
        true_xy = np.array([[c, -s], [s, c]]) @ p_odom[:2] + drift
        f.on_landmark_fix(true_xy, 0.05, yaw_fix=dyaw, yaw_odom=0.0)
    p, yaw = f.to_map(p_odom, 0.0)
    assert np.linalg.norm(p[:2] - true_xy) < 0.05
    # a random-walk filter lags a steadily growing drift slightly
    assert abs(yaw - dyaw) < math.radians(1.0)


def test_fusion_reanchors_on_vio_reset():
    from gdnav.bridge.landmark_vio_fusion import LandmarkVioFusion

    f = LandmarkVioFusion()
    f.on_odometry(np.array([5.0, 0.0, -2.0]), 0.0, reset_counter=0)
    # VIO restarts at its origin while the flight controller says we are at (8, 1)
    p, _ = f.on_odometry(np.zeros(3), 0.0, reset_counter=1, anchor=(np.array([8.0, 1.0, -2.0]), 0.0, 0.1))
    assert np.allclose(p[:2], [8.0, 1.0])
