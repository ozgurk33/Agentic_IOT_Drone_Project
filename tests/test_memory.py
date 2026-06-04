"""Tests for the short-term memory: telemetry history + tilt-trend summary."""
import pytest

from backend import config
from backend.agent_brain.agent_core import build_prompt, summarise_trend
from backend.state import DroneState

pytestmark = pytest.mark.usefixtures("fresh_state")


# ── State history ─────────────────────────────────────────────────────────────

def test_telemetry_history_is_capped():
    s = DroneState()
    for i in range(config.AGENT_MEMORY_FRAMES + 10):
        s.note_telemetry({"beta": float(i), "gamma": 0.0})
    assert len(s.telemetry_history) == config.AGENT_MEMORY_FRAMES
    # The most recent sample is the latest telemetry.
    assert s.latest_telemetry["beta"] == float(config.AGENT_MEMORY_FRAMES + 9)


def test_note_telemetry_ignores_empty_dropouts():
    s = DroneState()
    s.note_telemetry({"beta": 5.0})
    s.note_telemetry({})  # sensor dropout / bad packet
    assert len(s.telemetry_history) == 1   # dropout not stored in history
    assert s.latest_telemetry == {}        # but it IS the latest known reading


# ── Trend summary ─────────────────────────────────────────────────────────────

def test_summarise_trend_none_below_two_samples():
    assert summarise_trend([]) is None
    assert summarise_trend([{"beta": 5}]) is None


def test_summarise_trend_detects_rising():
    out = summarise_trend([{"beta": 5}, {"beta": 12}, {"beta": 20}])
    assert out is not None
    assert "RISING" in out
    assert "TILT_TREND" in out


def test_summarise_trend_detects_falling():
    out = summarise_trend([{"beta": 30}, {"beta": 18}, {"beta": 5}])
    assert "FALLING" in out


def test_summarise_trend_detects_steady():
    out = summarise_trend([{"beta": 10.0}, {"beta": 10.2}, {"beta": 9.8}])
    assert "STEADY" in out


# ── Prompt integration ────────────────────────────────────────────────────────

def test_build_prompt_includes_trend_when_history_present():
    history = [{"beta": 5}, {"beta": 12}, {"beta": 20}]
    parts, _ = build_prompt({"beta": 20, "gamma": 0}, None, use_vision=False, history=history)
    assert "TILT_TREND" in parts[0]
    assert "TILT_DEG=20.0" in parts[0]


def test_build_prompt_omits_trend_without_history():
    parts, _ = build_prompt({"beta": 20, "gamma": 0}, None, use_vision=False)
    assert "TILT_TREND" not in parts[0]
