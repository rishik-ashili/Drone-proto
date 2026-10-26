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
