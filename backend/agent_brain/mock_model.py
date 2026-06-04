"""
Mock LLM model — offline / CI substitute for Ollama
===================================================
A Pydantic-AI ``FunctionModel`` that emulates a competent vision-language flight
controller *without* a running Ollama server. It parses the ``TILT_DEG=`` value
that ``agent_core.build_prompt`` embeds in every prompt and returns a
telemetry-appropriate ``DroneDecision`` through the agent's structured-output
tool.

Why this exists:
  * The benchmark and the pytest suite must run end-to-end on machines that have
    no GPU / no Ollama (e.g. CI, the build box).
  * It lets us validate the *whole* pipeline — prompt assembly, structured-output
    parsing, MCP dispatch, metrics — independently of model quality.

A small artificial latency makes the latency columns in the benchmark non-zero
and visually distinct from the (much faster) rule engine.
"""
from __future__ import annotations

import random
import re
import time
from typing import Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from backend import config

_TILT_RE = re.compile(r"TILT_DEG=([-\d.]+)")


def _extract_tilt(messages: list[ModelMessage]) -> float:
    """Pull the TILT_DEG value out of the most recent user prompt."""
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            content = getattr(part, "content", None)
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, (list, tuple)):
                texts.extend(c for c in content if isinstance(c, str))
            for text in texts:
                m = _TILT_RE.search(text)
                if m:
                    try:
                        return abs(float(m.group(1)))
                    except ValueError:
                        return 0.0
    return 0.0


def _decide_args(tilt: float) -> dict:
    """Heuristic decision mirroring the safety policy in the system prompt."""
    if tilt >= config.TILT_DANGER_DEG:
        return {
            "action": "STABILIZE",
            "hazard_level": "CRITICAL" if tilt >= config.TILT_DANGER_DEG * 1.5 else "HIGH",
            "reasoning": f"Tilt {tilt:.1f}° is at/above the {config.TILT_DANGER_DEG:.0f}° "
            "danger threshold; equalising motors to recover attitude.",
            "confidence": 0.9,
        }
    if tilt >= config.TILT_WARNING_DEG:
        return {
            "action": "STABILIZE",
            "hazard_level": "MEDIUM",
            "reasoning": f"Tilt {tilt:.1f}° exceeds the {config.TILT_WARNING_DEG:.0f}° "
            "warning threshold; applying corrective stabilisation.",
            "confidence": 0.82,
        }
    return {
        "action": "HOVER",
        "hazard_level": "NONE",
        "reasoning": f"Tilt {tilt:.1f}° is within nominal limits; maintaining a "
        "balanced hover.",
        "confidence": 0.88,
    }


def build_mock_model(
    *,
    min_latency_ms: float = 40.0,
    max_latency_ms: float = 180.0,
    seed: Optional[int] = None,
) -> FunctionModel:
    """
    Build a FunctionModel that returns a telemetry-aware DroneDecision.

    Args:
        min_latency_ms / max_latency_ms: simulated "thinking time" injected per
            call so benchmark latency numbers look realistic.
        seed: optional RNG seed for reproducible latency.
    """
    rng = random.Random(seed)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Simulate inference cost (kept modest so test suites stay fast).
        time.sleep(rng.uniform(min_latency_ms, max_latency_ms) / 1000.0)

        tilt = _extract_tilt(messages)
        args = _decide_args(tilt)

        # Route the answer through the agent's structured-output tool.
        if info.output_tools:
            tool_name = info.output_tools[0].name
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

        # PromptedOutput path (no output tool registered): return JSON text that
        # pydantic-ai parses into the DroneDecision — mirrors how a real local
        # vision model answers when tool-calling isn't available.
        import json

        return ModelResponse(parts=[TextPart(content=json.dumps(args))])

    return FunctionModel(respond, model_name="mock-vision-controller")
