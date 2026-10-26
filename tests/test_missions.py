"""End-to-end missions in the built-in simulator (the plan's exit criteria)."""
from gdnav.navigation.state_machine import State
from gdnav.runner import build_sim_runner


def fly(mission, seed=1, **kw):
    r = build_sim_runner(mission, log_path=None, annotate=False, seed=seed, **kw)
    r.start()
    visited = set()
    while r.sm.state not in (State.LANDED, State.ABORTED) and r.t < 300:
        r.tick()
        visited.add(r.sm.state)
    s = r.finish()
    return r, s, visited


def test_p0_hello_square_reach_error():
    r, s, _ = fly("hello_square")
    assert r.sm.state == State.LANDED
    assert s["waypoints_done"] == 4
    assert s["reach_error_max_m"] < 0.5


def test_landmark_mission_end_to_end():
    r, s, visited = fly("investor_demo")
    assert r.sm.state == State.LANDED
    assert s["landmarks_found"] == 4
    assert State.ALIGN in visited
    assert s["collisions"] == 0
    assert s["reach_error_mean_m"] < 0.2
    assert s["rts_error_m"] < 0.8


def test_persistent_camera_fault_holds_then_lands():
    r, s, visited = fly("investor_demo", fault_schedule=[("camera_covered", 13.5, None)])
    assert State.FAILSAFE_HOLD in visited
    assert r.sm.state == State.LANDED
    assert s["failsafe_responses"] and s["failsafe_responses"][0]["response_time_s"] < 1.0


def test_low_battery_returns_to_start():
    r, s, visited = fly("investor_demo", fault_schedule=[("low_battery", 20.0, None)])
    assert State.RETURN_TO_START in visited
    assert r.sm.state == State.LANDED
