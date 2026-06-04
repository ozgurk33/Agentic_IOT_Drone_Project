"""Tests for the hybrid controller: reflex layer + arbitration + hybrid_step."""
import pytest

from backend import config
from backend.agent_brain.agent_core import (
    DroneAction,
    DroneDecision,
    HazardLevel,
    arbitrate,
    hybrid_step,
    reflex_decision,
)

pytestmark = pytest.mark.usefixtures("fresh_state")


def _mid_warning() -> float:
    """A tilt squarely inside the WARNING band (between warning and danger)."""
    return (config.TILT_WARNING_DEG + config.TILT_DANGER_DEG) / 2


# ── Reflex layer ──────────────────────────────────────────────────────────────

def test_reflex_decision_matches_thresholds():
    danger = reflex_decision({"beta": config.TILT_DANGER_DEG + 1, "gamma": 0})
    assert danger.action is DroneAction.STABILIZE
    assert danger.hazard_level is HazardLevel.CRITICAL
    assert danger.confidence == 1.0  # deterministic → fully certain

    warn = reflex_decision({"beta": _mid_warning(), "gamma": 0})
    assert warn.action is DroneAction.STABILIZE
    assert warn.hazard_level is HazardLevel.MEDIUM

    nominal = reflex_decision({"beta": 2.0, "gamma": 1.0})
    assert nominal.action is DroneAction.HOVER
    assert nominal.hazard_level is HazardLevel.NONE


# ── Arbitration (pure logic) ──────────────────────────────────────────────────

def test_arbitrate_danger_reflex_overrides():
    tilt = config.TILT_DANGER_DEG + 5
    reflex = reflex_decision({"beta": tilt})
    # Even if the agent thinks everything is fine, safety wins in danger.
    deliberate = DroneDecision(action=DroneAction.HOVER, reasoning="agent says fine")
    chosen, source = arbitrate(reflex, deliberate, tilt)
    assert source == "reflex-override"
    assert chosen.action is DroneAction.STABILIZE


def test_arbitrate_safe_uses_deliberation():
    tilt = 3.0
    reflex = reflex_decision({"beta": tilt})  # HOVER (nominal)
    deliberate = DroneDecision(
        action=DroneAction.TURN_LEFT, reasoning="obstacle on the right", turn_degrees=20
    )
    chosen, source = arbitrate(reflex, deliberate, tilt)
    assert source == "deliberation"
    assert chosen.action is DroneAction.TURN_LEFT


def test_arbitrate_reflex_floor_blocks_unsafe_relaxation():
    tilt = _mid_warning()
    reflex = reflex_decision({"beta": tilt})  # STABILIZE (warning band)
    deliberate = DroneDecision(action=DroneAction.HOVER, reasoning="agent under-reacts")
    chosen, source = arbitrate(reflex, deliberate, tilt)
    assert source == "reflex-floor"
    assert chosen.action is DroneAction.STABILIZE


def test_arbitrate_no_deliberation_uses_reflex():
    tilt = 3.0
    reflex = reflex_decision({"beta": tilt})
    chosen, source = arbitrate(reflex, None, tilt)
    assert source == "reflex-only"
    assert chosen is reflex


# ── Full hybrid step (offline mock agent) ─────────────────────────────────────

async def test_hybrid_step_danger_is_fast_and_safe(fresh_state, mock_agent):
    from backend.metrics import MetricsCollector

    coll = MetricsCollector(file_logging=False)
    fresh_state.latest_telemetry = {"beta": config.TILT_DANGER_DEG + 6, "gamma": 0.0}
    rec = await hybrid_step(mock_agent, scenario="unit", collector=coll)
    assert rec.mode == "hybrid"
    assert rec.action == "STABILIZE"
    # Reflex handled it without consulting the LLM → zero decision latency.
    assert rec.latency_ms == 0.0
    assert "reflex-override" in rec.reasoning
    assert fresh_state.motors.mode == "stabilize"


async def test_hybrid_step_safe_uses_agent(fresh_state, mock_agent):
    from backend.metrics import MetricsCollector

    coll = MetricsCollector(file_logging=False)
    fresh_state.latest_telemetry = {"beta": 2.0, "gamma": 1.0}
    rec = await hybrid_step(mock_agent, collector=coll)
    assert rec.mode == "hybrid"
    assert rec.action == "HOVER"
    assert "deliberation" in rec.reasoning


async def test_hybrid_step_survives_dead_model(fresh_state):
    """A broken agent must not crash hybrid_step — the reflex keeps it safe."""
    from backend.agent_brain.agent_core import build_agent
    from backend.metrics import MetricsCollector
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    dead = OllamaModel(
        config.OLLAMA_MODEL,
        provider=OllamaProvider(base_url="http://127.0.0.1:9/v1"),
    )
    agent = build_agent(dead)
    coll = MetricsCollector(file_logging=False)
    # Safe tilt: the agent WOULD be consulted, but it's dead → reflex-only.
    fresh_state.latest_telemetry = {"beta": 2.0, "gamma": 1.0}
    rec = await hybrid_step(agent, collector=coll)
    assert rec.ok is False
    assert rec.error is not None
    assert rec.action == "HOVER"          # reflex safe action still applied
    assert "reflex-only" in rec.reasoning


# ── Live reflex tick (the 10 Hz path used by hybrid_controller_loop) ──────────

async def test_hybrid_reflex_tick_applies_fresh_deliberation(fresh_state, monkeypatch):
    import time

    import backend.agent_brain.agent_core as ac
    from backend.metrics import MetricsCollector

    monkeypatch.setattr(ac, "metrics", MetricsCollector(file_logging=False))
    fresh_state.latest_telemetry = {"beta": 3.0, "gamma": 1.0}      # safe band
    fresh_state.hybrid_deliberation = {
        "action": "TURN_LEFT", "hazard": "LOW", "reasoning": "clear path to the left",
        "confidence": 0.8, "target_motor_speed": None, "turn_degrees": 20.0,
        "latency_ms": 1234.0, "used_vision": True, "ts": time.time(),
    }
    key = await ac._hybrid_reflex_tick(None)
    assert key == ("TURN_LEFT", "deliberation")
    assert fresh_state.motors.mode == "turn_left"   # the agent's verdict was applied


async def test_hybrid_reflex_tick_danger_overrides_cache(fresh_state, monkeypatch):
    import time

    import backend.agent_brain.agent_core as ac
    from backend.metrics import MetricsCollector

    monkeypatch.setattr(ac, "metrics", MetricsCollector(file_logging=False))
    # Danger tilt: even a fresh "everything's fine" cache must be ignored.
    fresh_state.latest_telemetry = {"beta": config.TILT_DANGER_DEG + 5, "gamma": 0.0}
    fresh_state.hybrid_deliberation = {
        "action": "HOVER", "hazard": "NONE", "reasoning": "looks fine",
        "confidence": 0.9, "ts": time.time(),
    }
    key = await ac._hybrid_reflex_tick(None)
    assert key == ("STABILIZE", "reflex-override")
    assert fresh_state.motors.mode == "stabilize"


async def test_hybrid_reflex_tick_stale_cache_falls_back_to_reflex(fresh_state, monkeypatch):
    import time

    import backend.agent_brain.agent_core as ac
    from backend.metrics import MetricsCollector

    monkeypatch.setattr(ac, "metrics", MetricsCollector(file_logging=False))
    fresh_state.latest_telemetry = {"beta": 3.0, "gamma": 1.0}      # safe band
    # Deliberation older than the TTL → treated as stale → reflex-only.
    fresh_state.hybrid_deliberation = {
        "action": "TURN_LEFT", "hazard": "LOW", "reasoning": "old idea",
        "confidence": 0.8, "turn_degrees": 20.0,
        "ts": time.time() - (config.HYBRID_DELIBERATION_TTL_S + 5.0),
    }
    key = await ac._hybrid_reflex_tick(None)
    assert key == ("HOVER", "reflex-only")
