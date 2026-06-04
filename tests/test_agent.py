"""Tests for the Pydantic-AI agentic controller (using the offline mock model)."""
import pytest

from backend import config
from backend.agent_brain.agent_core import (
    DroneAction,
    DroneDecision,
    HazardLevel,
    agent_step,
    build_prompt,
    compute_tilt,
    decide,
    dispatch_decision,
    safe_fallback_decision,
)

pytestmark = pytest.mark.usefixtures("fresh_state")


def test_compute_tilt_worst_axis():
    assert compute_tilt({"beta": 10, "gamma": 25}) == 25
    assert compute_tilt({"beta": -40, "gamma": 5}) == 40
    assert compute_tilt({}) == 0


def test_build_prompt_embeds_tilt_marker():
    parts, used_vision = build_prompt({"beta": 12.5, "gamma": 3.0}, None, use_vision=True)
    assert isinstance(parts[0], str)
    assert "TILT_DEG=12.5" in parts[0]
    assert used_vision is False  # no frame supplied


def test_build_prompt_attaches_frame():
    from backend.scenarios import render_frame_b64
    from pydantic_ai import BinaryContent

    frame = render_frame_b64({"beta": 5, "gamma": 5})
    if frame is None:
        pytest.skip("Pillow not available")
    parts, used_vision = build_prompt({"beta": 5, "gamma": 5}, frame, use_vision=True)
    assert used_vision is True
    assert any(isinstance(p, BinaryContent) for p in parts)


async def test_agent_danger_stabilizes(fresh_state, mock_agent):
    tele = {"beta": config.TILT_DANGER_DEG + 6, "gamma": 1.0}
    decision, latency_ms, _ = await decide(mock_agent, tele, None, use_vision=False)
    assert decision.action is DroneAction.STABILIZE
    assert decision.hazard_level in (HazardLevel.HIGH, HazardLevel.CRITICAL)
    assert latency_ms >= 0


async def test_agent_nominal_hovers(fresh_state, mock_agent):
    decision, _, _ = await decide(mock_agent, {"beta": 2.0, "gamma": 1.0}, None, use_vision=False)
    assert decision.action is DroneAction.HOVER
    assert decision.hazard_level is HazardLevel.NONE


async def test_agent_step_records_and_applies(fresh_state, mock_agent):
    from backend.metrics import MetricsCollector

    coll = MetricsCollector(file_logging=False)
    fresh_state.latest_telemetry = {"beta": config.TILT_DANGER_DEG + 2, "gamma": 0.0}
    rec = await agent_step(mock_agent, scenario="unit", collector=coll)
    assert rec.mode == "agentic"
    assert rec.action == "STABILIZE"
    assert rec.scenario == "unit"
    assert rec.ok is True
    # Dispatch must have moved the motors.
    assert fresh_state.motors.mode == "stabilize"
    assert coll.recent(10)[-1]["action"] == "STABILIZE"


async def test_dispatch_all_actions(fresh_state):
    res = await dispatch_decision(DroneDecision(action=DroneAction.EMERGENCY_STOP, reasoning="x"))
    assert res["action"] == "emergency_stop"
    assert fresh_state.motors.fl == 0

    res = await dispatch_decision(DroneDecision(
        action=DroneAction.ADJUST_MOTOR, reasoning="x", target_motor_speed=42))
    assert fresh_state.motors.fl == 42

    res = await dispatch_decision(DroneDecision(
        action=DroneAction.TURN_LEFT, reasoning="x", turn_degrees=20))
    assert fresh_state.motors.mode == "turn_left"

    res = await dispatch_decision(DroneDecision(action=DroneAction.HOLD, reasoning="x"))
    assert res["ok"] is True


async def test_adjust_motor_without_value_is_safe(fresh_state):
    res = await dispatch_decision(DroneDecision(action=DroneAction.ADJUST_MOTOR, reasoning="x"))
    assert res["ok"] is False  # refuses rather than guessing


def test_safe_fallback_matches_rule_thresholds():
    danger = safe_fallback_decision({"beta": config.TILT_DANGER_DEG + 1, "gamma": 0}, "down")
    assert danger.action is DroneAction.STABILIZE
    assert danger.confidence == 0.0
    nominal = safe_fallback_decision({"beta": 1, "gamma": 1}, "down")
    assert nominal.action is DroneAction.HOVER


async def test_agent_step_handles_llm_failure(fresh_state):
    """A broken model must not crash agent_step — it falls back safely."""
    from backend.agent_brain.agent_core import build_agent
    from backend.metrics import MetricsCollector
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    # Ollama model pointed at a guaranteed-dead port → connection error inside
    # decide(). Deterministic regardless of whether a real Ollama is running.
    dead = OllamaModel(
        config.OLLAMA_MODEL,
        provider=OllamaProvider(base_url="http://127.0.0.1:9/v1"),
    )
    agent = build_agent(dead)
    coll = MetricsCollector(file_logging=False)
    fresh_state.latest_telemetry = {"beta": config.TILT_DANGER_DEG + 5, "gamma": 0}
    rec = await agent_step(agent, collector=coll)
    assert rec.ok is False
    assert rec.error is not None
    # Even on failure it applied a safe stabilising action.
    assert rec.action == "STABILIZE"
