"""
Shared in-process state singleton.
Imported by main.py, mcp_server.py, and agent_core.py.
Extracted here so none of those modules form a circular import.
"""
import asyncio
import time
from collections import deque
from typing import Deque, Optional

from backend import config


class MotorState:
    """Virtual quadrotor motor speeds (0–100 %) and current flight mode."""

    MOTORS = ("fl", "fr", "rl", "rr")  # front-left, front-right, rear-left, rear-right

    def __init__(self) -> None:
        self.fl: int = config.MOTOR_DEFAULT
        self.fr: int = config.MOTOR_DEFAULT
        self.rl: int = config.MOTOR_DEFAULT
        self.rr: int = config.MOTOR_DEFAULT
        self.mode: str = "hover"

    def set_all(self, speed: int, mode: str = "hover") -> None:
        speed = max(0, min(100, int(speed)))
        self.fl = self.fr = self.rl = self.rr = speed
        self.mode = mode

    def to_dict(self) -> dict:
        return {
            "fl": self.fl,
            "fr": self.fr,
            "rl": self.rl,
            "rr": self.rr,
            "mode": self.mode,
        }


class DroneState:
    """
    Central mutable state shared by the FastAPI server, MCP server, and agent.

    Thread/asyncio safety: all mutations happen inside the single asyncio
    event loop — no lock needed.

    frame_queue has maxsize=1 (backpressure): the agent always reads the
    *latest* frame, never a stale frame queued while inference was running.
    """

    def __init__(self) -> None:
        self.latest_frame_b64: Optional[str] = None
        self.latest_telemetry: dict = {}
        self.agent_mode: str = "traditional"  # "traditional" | "agentic" | "hybrid"
        self.decision_log: list[dict] = []
        self.drone_connected: bool = False
        self.dashboards_connected: int = 0
        self.motors: MotorState = MotorState()

        # Short-term memory: the last N telemetry samples, so the agent can reason
        # about how the attitude is *changing* (rising/falling tilt), not just the
        # current instant. Bounded — old samples fall off automatically.
        self.telemetry_history: Deque[dict] = deque(maxlen=max(1, config.AGENT_MEMORY_FRAMES))

        # Latest agent verdict in hybrid mode, cached by the slow deliberation
        # layer and consumed by the fast reflex layer. Plain dict (not a
        # DroneDecision) to keep this module free of any agent imports.
        self.hybrid_deliberation: Optional[dict] = None

        # Shadow decisions: both rule and agent run in the background regardless
        # of active mode, storing their latest verdict here.  The dashboard
        # compare-strip reads these to show both modes side-by-side at all times.
        self.shadow_decisions: dict = {"rule": None, "agent": None, "nav": None}

        # Lazy so it's created inside the running event loop (Python 3.10+ safe)
        self._frame_queue: Optional[asyncio.Queue] = None

    @property
    def frame_queue(self) -> asyncio.Queue:
        if self._frame_queue is None:
            self._frame_queue = asyncio.Queue(maxsize=1)
        return self._frame_queue

    # ── Telemetry / frame ingestion ─────────────────────────────────────────────

    def note_telemetry(self, telemetry: dict) -> None:
        """
        Record a fresh telemetry sample: set it as the latest AND append it to the
        short-term history. Use this (instead of assigning ``latest_telemetry``
        directly) so the rolling memory stays in sync. Empty dicts (sensor
        dropouts) update ``latest_telemetry`` but are not added to the history.
        """
        self.latest_telemetry = telemetry
        if telemetry:
            self.telemetry_history.append(telemetry)

    def push_frame(self, frame_b64: str, telemetry: dict) -> None:
        """Store latest frame; evict any unconsumed queued frame first."""
        self.latest_frame_b64 = frame_b64
        self.latest_telemetry = telemetry
        q = self.frame_queue
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait((frame_b64, telemetry))
        except asyncio.QueueFull:
            pass

    # ── Logging ───────────────────────────────────────────────────────────────

    def add_log(
        self,
        source: str,
        message: str,
        level: str = "INFO",
        extra: Optional[dict] = None,
    ) -> None:
        entry: dict = {
            "ts": time.time(),
            "source": source,
            "message": message,
            "level": level,
        }
        if extra:
            entry["extra"] = extra
        self.decision_log.append(entry)
        if len(self.decision_log) > config.MAX_DECISION_LOG:
            self.decision_log = self.decision_log[-config.MAX_DECISION_LOG :]


# ── Module-level singleton ────────────────────────────────────────────────────
drone_state = DroneState()
