"""Tests for the traditional IF/THEN rule engine."""
import pytest

from backend import config
from backend.mcp_server.mcp_server import run_traditional_rules


pytestmark = pytest.mark.usefixtures("fresh_state")


async def test_no_telemetry_holds(fresh_state):
    fresh_state.latest_telemetry = {}
    result = await run_traditional_rules()
    assert result["rule"] == "NO_DATA"
    assert result["action"] == "hold"


async def test_nominal_hover(fresh_state):
    fresh_state.latest_telemetry = {"beta": 5.0, "gamma": 3.0}
    result = await run_traditional_rules()
    assert result["rule"] == "HOVER"
    assert result["motor_speed"] == config.MOTOR_DEFAULT
    assert fresh_state.motors.mode == "hover"


async def test_warning_stabilize(fresh_state):
    tilt = (config.TILT_WARNING_DEG + config.TILT_DANGER_DEG) / 2
    fresh_state.latest_telemetry = {"beta": tilt, "gamma": 0.0}
    result = await run_traditional_rules()
    assert result["rule"] == "STABILIZE"
    assert result["motor_speed"] == config.MOTOR_STABILIZE


async def test_danger_emergency_stabilize(fresh_state):
    fresh_state.latest_telemetry = {"beta": config.TILT_DANGER_DEG + 5, "gamma": 0.0}
    result = await run_traditional_rules()
    assert result["rule"] == "EMERGENCY_STABILIZE"
    assert result["motor_speed"] == config.MOTOR_EMERGENCY
    assert fresh_state.motors.fl == config.MOTOR_EMERGENCY


async def test_worst_axis_tilt_uses_max(fresh_state):
    # roll (gamma) is the larger axis here and must drive the decision.
    fresh_state.latest_telemetry = {"beta": 2.0, "gamma": config.TILT_DANGER_DEG + 1}
    result = await run_traditional_rules()
    assert result["rule"] == "EMERGENCY_STABILIZE"
