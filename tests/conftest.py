"""Shared pytest fixtures."""

import pytest

from backend.state import DroneState
import backend.state as state_module


@pytest.fixture
def fresh_state(monkeypatch):
    """
    Replace the global drone_state singleton with a clean instance for the test,
    so tests never leak motor/log state into one another.
    """
    new_state = DroneState()
    monkeypatch.setattr(state_module, "drone_state", new_state)
    # Modules that imported the singleton by reference need patching too.
    import backend.mcp_server.mcp_server as mcp_mod
    import backend.agent_brain.agent_core as agent_mod
    monkeypatch.setattr(mcp_mod, "drone_state", new_state)
    monkeypatch.setattr(agent_mod, "drone_state", new_state)
    return new_state


@pytest.fixture
def mock_agent():
    from backend.agent_brain.agent_core import build_agent
    from backend.agent_brain.mock_model import build_mock_model
    # Near-zero latency so the suite stays fast.
    return build_agent(build_mock_model(min_latency_ms=0.0, max_latency_ms=1.0, seed=7))
