"""Tests for synthetic scenarios and the shared DroneState."""
import asyncio

import pytest

from backend import config
from backend import scenarios as scen
from backend.state import DroneState, MotorState


# ── Scenarios ───────────────────────────────────────────────────────────────

def test_all_scenarios_build_and_are_nonempty():
    sc = scen.all_scenarios()
    assert len(sc) == len(scen.ALL_BUILDERS)
    for s in sc:
        assert len(s) > 0
        for step in s.steps:
            assert isinstance(step, dict)  # may be {} for dropouts


def test_gradual_tilt_crosses_thresholds():
    s = scen.get_scenario("gradual_tilt")
    tilts = [max(abs(st.get("beta", 0)), abs(st.get("gamma", 0))) for st in s.steps]
    assert min(tilts) < config.TILT_WARNING_DEG
    assert max(tilts) >= config.TILT_DANGER_DEG


def test_sensor_dropout_has_empty_steps():
    s = scen.get_scenario("sensor_dropout")
    assert any(st == {} for st in s.steps)


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        scen.get_scenario("does_not_exist")


def test_render_frame_returns_base64():
    frame = scen.render_frame_b64({"beta": 10, "gamma": 20})
    if frame is None:
        pytest.skip("Pillow not available")
    import base64
    raw = base64.b64decode(frame)
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes


# ── DroneState ──────────────────────────────────────────────────────────────

def test_motor_clamping():
    m = MotorState()
    m.set_all(250)
    assert m.fl == 100
    m.set_all(-10)
    assert m.fl == 0


def test_log_buffer_capped():
    s = DroneState()
    for i in range(config.MAX_DECISION_LOG + 50):
        s.add_log("T", f"msg {i}")
    assert len(s.decision_log) == config.MAX_DECISION_LOG
    assert s.decision_log[-1]["message"] == f"msg {config.MAX_DECISION_LOG + 49}"


def test_frame_queue_backpressure():
    async def run():
        s = DroneState()
        s.push_frame("frame1", {"a": 1})
        s.push_frame("frame2", {"a": 2})  # should evict frame1
        assert s.frame_queue.qsize() == 1
        frame, tele = s.frame_queue.get_nowait()
        assert frame == "frame2"
        assert s.latest_frame_b64 == "frame2"

    asyncio.run(run())
