"""
Decision metrics & structured logging
======================================
A single place where *both* control modes record what they decided and how long
it took.  Two sinks:

  1. An in-memory ring buffer (``MetricsCollector.records``) exposed via
     ``/api/metrics`` and consumed by the Phase-5 benchmark.
  2. An append-only JSON-Lines file (one JSON object per line) so a full run can
     be replayed / analysed offline.

Keeping the schema identical for the rule engine and the agent is what makes the
"Rule-Based vs Agentic" comparison in the jury presentation an apples-to-apples
one.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Optional

from backend import config


@dataclass
class DecisionRecord:
    """One control decision — emitted by the rule engine or the agent."""

    ts: float                       # Unix time the decision was finalised
    mode: str                       # "traditional" | "agentic"
    action: str                     # canonical action name (HOVER, STABILIZE, ...)
    latency_ms: float               # wall-clock time spent producing the decision
    tilt_deg: Optional[float] = None
    hazard_level: Optional[str] = None
    motor_speed: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: str = ""             # human-readable explanation (agent) / rule id
    used_vision: bool = False
    ok: bool = True                 # False when the decision came from a fallback
    error: Optional[str] = None
    scenario: Optional[str] = None  # set by the benchmark harness

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCollector:
    """Thread-safe ring buffer + optional JSONL sink."""

    def __init__(
        self,
        maxlen: int = config.METRICS_BUFFER_SIZE,
        jsonl_path: Optional[str] = None,
        file_logging: bool = config.ENABLE_FILE_LOGGING,
    ) -> None:
        self._records: Deque[DecisionRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._file_logging = file_logging
        self._jsonl_path = Path(jsonl_path or config.DECISION_LOG_FILE)
        if self._file_logging:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, rec: DecisionRecord) -> DecisionRecord:
        with self._lock:
            self._records.append(rec)
            if self._file_logging:
                try:
                    with self._jsonl_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
                except OSError:
                    # Never let logging break the control loop.
                    pass
        return rec

    # ── reads ─────────────────────────────────────────────────────────────────

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._records)[-limit:]
        return [r.to_dict() for r in items]

    def summary(self) -> dict:
        """Aggregate stats split by mode — the numbers the jury cares about."""
        with self._lock:
            items = list(self._records)
        return summarise(items)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


def summarise(records: list[DecisionRecord]) -> dict:
    """Compute per-mode aggregates (count, latency p50/p95/avg, action mix)."""
    by_mode: dict[str, list[DecisionRecord]] = {}
    for r in records:
        by_mode.setdefault(r.mode, []).append(r)

    out: dict = {"total": len(records), "by_mode": {}}
    for mode, recs in by_mode.items():
        lat = sorted(r.latency_ms for r in recs)
        actions: dict[str, int] = {}
        for r in recs:
            actions[r.action] = actions.get(r.action, 0) + 1
        confs = [r.confidence for r in recs if r.confidence is not None]
        out["by_mode"][mode] = {
            "count": len(recs),
            "latency_ms": {
                "avg": round(sum(lat) / len(lat), 2) if lat else 0.0,
                "min": round(lat[0], 2) if lat else 0.0,
                "max": round(lat[-1], 2) if lat else 0.0,
                "p50": round(_percentile(lat, 50), 2) if lat else 0.0,
                "p95": round(_percentile(lat, 95), 2) if lat else 0.0,
            },
            "actions": actions,
            "errors": sum(1 for r in recs if not r.ok),
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        }
    return out


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ── Module-level singleton used by the live server ──────────────────────────────
metrics = MetricsCollector()


def stopwatch() -> "Timer":
    return Timer()


@dataclass
class Timer:
    """Tiny context-manager stopwatch: ``with stopwatch() as t: ...; t.ms``."""

    _start: float = field(default_factory=time.perf_counter)
    ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
