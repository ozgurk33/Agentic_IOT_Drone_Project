"""
Pydantic-AI Agent Core — Phase 4
================================
The "agentic" brain of the drone.  Where the Phase-3 rule engine is a fixed
IF/THEN ladder, this module hands the *same* sensor inputs to a local LLM
(Ollama, vision-capable) and lets it decide what to do — then routes that
decision through the Phase-3 MCP tool layer.

Pipeline for one tick (``agent_step``):

    latest JPEG frame  ─┐
                        ├─►  Pydantic-AI Agent (Ollama)  ─►  DroneDecision
    IMU telemetry      ─┘                                         │
                                                                  ▼
                                          dispatch_decision()  →  MCP tools
                                                                  │
                                                                  ▼
                                            DroneState.motors / decision log
                                                                  │
                                                                  ▼
                                              MetricsCollector (latency, action)

Design goals:
  * **Robust** — Ollama unreachable, a malformed model reply, or a slow tick
    must never crash the loop; the drone falls back to a safe HOLD/STABILIZE.
  * **Explainable** — every decision carries a free-text ``reasoning`` string
    that is surfaced on the dashboard and written to the JSONL log.
  * **Comparable** — emits the exact same ``DecisionRecord`` schema the rule
    engine uses, so the benchmark can line them up side by side.
"""
from __future__ import annotations

import base64
import binascii
import enum
import io
import logging
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput, PromptedOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from backend import config
from backend.metrics import DecisionRecord, MetricsCollector, metrics, stopwatch
from backend.mcp_server import mcp_server as tools
from backend.state import drone_state

logger = logging.getLogger("drone.agent")


# ─────────────────────────────────────────────────────────────────────────────
# Decision schema — the structured output the LLM must produce
# ─────────────────────────────────────────────────────────────────────────────

class DroneAction(str, enum.Enum):
    HOVER = "HOVER"                  # balanced hover at default speed
    STABILIZE = "STABILIZE"          # counter moderate tilt
    EMERGENCY_STOP = "EMERGENCY_STOP"  # cut all power (last resort)
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    ADJUST_MOTOR = "ADJUST_MOTOR"    # set all motors to a specific speed
    HOLD = "HOLD"                    # keep current motor state, take no action


class HazardLevel(str, enum.Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DroneDecision(BaseModel):
    """The agent's structured verdict for a single control tick."""

    action: DroneAction = Field(
        description="The single control action to execute this tick."
    )
    hazard_level: HazardLevel = Field(
        default=HazardLevel.NONE,
        description="How dangerous the current situation is.",
    )
    reasoning: str = Field(
        description="One or two sentences explaining WHY this action was chosen, "
        "referencing the tilt/telemetry and (if used) what the camera shows.",
    )
    target_motor_speed: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Required only when action is ADJUST_MOTOR: the 0-100% speed "
        "to apply to all four motors.",
    )
    turn_degrees: Optional[float] = Field(
        default=None,
        ge=0,
        le=180,
        description="Required only when action is TURN_LEFT/TURN_RIGHT: how many "
        "degrees of yaw to command.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Self-rated confidence in this decision, 0.0–1.0.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# System / instruction prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Regenerate the system prompt with the current (possibly runtime-updated) thresholds."""
    return (
        "You are the autonomous flight controller of a simulated quadrotor drone whose "
        '"body" is a smartphone. Each tick you receive the drone\'s latest IMU telemetry '
        "(and, when available, a camera frame) and must choose exactly ONE control action.\n\n"
        "Coordinate / sensor conventions:\n"
        "  * tilt_deg = max(|pitch|, |roll|) — the worst-axis tilt of the airframe.\n"
        "  * A tilt of 0° is perfectly level. Larger is more dangerous.\n\n"
        "Safety policy (you must respect these thresholds):\n"
        f"  * tilt_deg >= {config.TILT_DANGER_DEG:.0f}°  → the drone is in DANGER. Choose\n"
        "    STABILIZE (preferred) or, only if the situation is unrecoverable,\n"
        "    EMERGENCY_STOP. Set hazard_level to HIGH or CRITICAL.\n"
        f"  * {config.TILT_WARNING_DEG:.0f}° <= tilt_deg < {config.TILT_DANGER_DEG:.0f}°  →\n"
        "    WARNING. Choose STABILIZE. hazard_level MEDIUM.\n"
        f"  * tilt_deg < {config.TILT_WARNING_DEG:.0f}°  → nominal. Choose HOVER (or a gentle\n"
        "    TURN / ADJUST_MOTOR if the camera shows an obstacle to avoid). hazard_level\n"
        "    NONE or LOW.\n\n"
        "Action contract:\n"
        "  * ADJUST_MOTOR requires target_motor_speed (0-100).\n"
        "  * TURN_LEFT / TURN_RIGHT require turn_degrees (0-180).\n"
        "  * Always fill 'reasoning' with a short, concrete justification.\n"
        "  * Prefer the least aggressive action that keeps the drone safe.\n\n"
        "WORKED EXAMPLES (follow this logic exactly):\n"
        f"  • tilt_deg=5,  TILT_TREND=STEADY  → HOVER. hazard NONE.\n"
        f"  • tilt_deg=12, TILT_TREND=STEADY  → HOVER (below threshold, stable). hazard LOW.\n"
        f"  • tilt_deg=12, TILT_TREND=RISING  → STABILIZE (below threshold BUT trending toward danger — act proactively). hazard MEDIUM.\n"
        f"  • tilt_deg=13, TILT_TREND=RISING  → STABILIZE (13° and climbing toward {config.TILT_WARNING_DEG:.0f}° — intervene now). hazard MEDIUM.\n"
        f"  • tilt_deg=20, TILT_TREND=STEADY  → STABILIZE (between {config.TILT_WARNING_DEG:.0f} and {config.TILT_DANGER_DEG:.0f}). hazard MEDIUM.\n"
        f"  • tilt_deg=35, TILT_TREND=RISING  → STABILIZE (>= {config.TILT_DANGER_DEG:.0f}, danger). hazard HIGH.\n"
        "  KEY INSIGHT: When TILT_TREND=RISING and tilt is approaching the warning threshold,\n"
        f"  choose STABILIZE proactively — do NOT wait for tilt to reach {config.TILT_WARNING_DEG:.0f}°.\n"
        "  This is your advantage over the rule engine: you can see the trend, it cannot.\n\n"
        "You are NOT a chatbot: respond only with the structured decision."
    )


# Keep the module-level name for backward-compat (tests import it).
SYSTEM_PROMPT = _build_system_prompt()


# ─────────────────────────────────────────────────────────────────────────────
# Model / agent construction
# ─────────────────────────────────────────────────────────────────────────────

def build_ollama_model() -> Model:
    """Construct the live Ollama vision model via its OpenAI-compatible endpoint."""
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    provider = OllamaProvider(base_url=config.OLLAMA_OPENAI_BASE_URL)
    return OllamaModel(config.OLLAMA_MODEL, provider=provider)


def _build_output_spec(mode: str):
    """Map a structured-output mode name to a pydantic-ai output spec."""
    if mode == "native":
        # Ollama constrains decoding to the JSON schema → reliable even with weak
        # local vision models. This is the right default for the live server.
        return NativeOutput(DroneDecision)
    if mode == "prompted":
        # Schema described in the prompt, model returns free JSON we parse.
        return PromptedOutput(DroneDecision)
    # "tool": the OpenAI tool-calling API (pydantic-ai default; used by the mock).
    return DroneDecision


def build_agent(
    model: Optional[Model | str] = None,
    *,
    output_mode: Optional[str] = None,
) -> Agent[None, DroneDecision]:
    """
    Build the Pydantic-AI agent.

    Args:
        model: Override the model — pass a ``Model`` instance (e.g. the mock model
               for tests/benchmarks) or a model name string. ``None`` builds the
               live Ollama model from config.
        output_mode: How to enforce the structured ``DroneDecision`` output —
               ``"native"`` (Ollama schema-constrained decoding),
               ``"prompted"`` (schema in prompt, parse the reply), or
               ``"tool"`` (OpenAI tool-calling; what the in-process mock emits).
               ``None`` (default) auto-selects: ``"native"`` for the live Ollama
               model, ``"tool"`` when an explicit model is supplied.

    Why this matters: local vision models served by Ollama (llava, llama3.2-vision,
    qwen2.5vl, …) do **not** support the OpenAI tool-calling API that pydantic-ai
    uses by default for structured output — they return HTTP 400 "does not support
    tools". Native (schema-constrained) output sidesteps that and, unlike prompted
    output, forces valid JSON even from small models that would otherwise echo the
    schema back instead of filling it in.
    """
    resolved: Model | str = model if model is not None else build_ollama_model()
    if output_mode is None:
        output_mode = "native" if model is None else "tool"
    return Agent(
        resolved,
        output_type=_build_output_spec(output_mode),
        instructions=_build_system_prompt(),  # always use current thresholds
        retries=config.AGENT_OUTPUT_RETRIES,
        model_settings=ModelSettings(
            temperature=0.1,
            timeout=config.OLLAMA_TIMEOUT_S,
        ),
        name="drone-flight-controller",
    )


# Lazily-built singleton for the live server loop.
_live_agent: Optional[Agent[None, DroneDecision]] = None


def get_live_agent() -> Agent[None, DroneDecision]:
    global _live_agent
    if _live_agent is None:
        _live_agent = build_agent()
    return _live_agent


def switch_model(model_name: str) -> None:
    """
    Hot-swap the Ollama model at runtime.
    Resets both the stabilization and navigation agent singletons so the next
    inference call rebuilds them with the new model.
    """
    global _live_agent
    config.OLLAMA_MODEL = model_name
    # Rebuild the OpenAI-compat URL (base URL unchanged, model name is in the API call)
    _live_agent = None
    # Nav agent reset is handled in nav_agent.py via its own reset function.


# ─────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────────────────────────────────────

def compute_tilt(telemetry: dict) -> float:
    """Worst-axis tilt — identical definition to the rule engine."""
    beta = abs(float(telemetry.get("beta", 0.0) or 0.0))
    gamma = abs(float(telemetry.get("gamma", 0.0) or 0.0))
    return max(beta, gamma)


def summarise_trend(history) -> Optional[str]:
    """
    Short-term memory → a one-line tilt trend for the prompt.

    Given the recent telemetry samples (``drone_state.telemetry_history``), report
    whether the worst-axis tilt is RISING, FALLING or STEADY and by how much. This
    gives the agent temporal context — a single frame can't tell it whether a 20°
    tilt is recovering or about to exceed the danger threshold.

    Returns ``None`` when fewer than two samples are available (nothing to trend).
    """
    tilts = [compute_tilt(t) for t in history if t]
    if len(tilts) < 2:
        return None
    recent = tilts[-config.AGENT_MEMORY_FRAMES:]
    delta = recent[-1] - recent[0]
    if delta > 1.0:
        trend = "RISING"
    elif delta < -1.0:
        trend = "FALLING"
    else:
        trend = "STEADY"
    arrow = "→".join(f"{v:.0f}" for v in recent)
    return (
        f"TILT_TREND (last {len(recent)} samples): {arrow}° "
        f"— {trend}, {delta:+.1f}° over the window."
    )


def _format_telemetry(telemetry: dict, trend: Optional[str] = None) -> str:
    tilt = compute_tilt(telemetry)
    trend_line = f"  {trend}\n" if trend else ""
    return (
        "DRONE TELEMETRY\n"
        f"  TILT_DEG={tilt:.1f}\n"
        f"  pitch(beta)={float(telemetry.get('beta', 0.0) or 0.0):.1f}  "
        f"roll(gamma)={float(telemetry.get('gamma', 0.0) or 0.0):.1f}  "
        f"yaw(alpha)={float(telemetry.get('alpha', 0.0) or 0.0):.1f}\n"
        f"  accel=({float(telemetry.get('ax', 0.0) or 0.0):.2f}, "
        f"{float(telemetry.get('ay', 0.0) or 0.0):.2f}, "
        f"{float(telemetry.get('az', 0.0) or 0.0):.2f}) m/s^2\n"
        f"  motors={drone_state.motors.to_dict()}\n"
        f"{trend_line}"
        "Decide the single best control action now."
    )


def _decode_frame(frame_b64: Optional[str]) -> Optional[bytes]:
    """Base64 → JPEG bytes, optionally down-scaled. None on any problem."""
    if not frame_b64:
        return None
    # Tolerate a data-URL prefix if one slips through.
    if frame_b64.startswith("data:"):
        frame_b64 = frame_b64.split(",", 1)[-1]
    try:
        raw = base64.b64decode(frame_b64)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None

    if config.AGENT_VISION_MAX_EDGE <= 0:
        return raw
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
        max_edge = config.AGENT_VISION_MAX_EDGE
        if max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=80)
        return out.getvalue()
    except Exception:
        # If PIL can't read it, fall back to the raw bytes — the model may cope.
        return raw


def build_prompt(
    telemetry: dict,
    frame_b64: Optional[str],
    use_vision: bool,
    history=None,
) -> tuple[list, bool]:
    """
    Return (prompt_parts, used_vision).

    prompt_parts is the list passed to ``agent.run`` — a text part plus, when a
    frame is available and vision is enabled, a BinaryContent image part. When a
    telemetry ``history`` is supplied, a one-line tilt trend (short-term memory)
    is folded into the text part.
    """
    trend = summarise_trend(history) if history else None
    parts: list = [_format_telemetry(telemetry, trend)]
    used_vision = False
    if use_vision and config.AGENT_USE_VISION:
        jpeg = _decode_frame(frame_b64)
        if jpeg:
            parts.append(BinaryContent(data=jpeg, media_type="image/jpeg"))
            used_vision = True
    return parts, used_vision


# ─────────────────────────────────────────────────────────────────────────────
# Decision dispatch — route the LLM's verdict through the MCP tool layer
# ─────────────────────────────────────────────────────────────────────────────

async def dispatch_decision(decision: DroneDecision) -> dict:
    """
    Execute ``decision`` by calling the corresponding Phase-3 MCP tool.

    Returns the tool's result dict. Unknown / malformed combinations degrade to a
    safe action rather than raising.
    """
    action = decision.action

    if action is DroneAction.EMERGENCY_STOP:
        return await tools.emergency_stop()

    if action is DroneAction.STABILIZE:
        return await tools.stabilize_drone()

    if action is DroneAction.HOVER:
        return await tools.set_hover()

    if action is DroneAction.TURN_LEFT:
        return await tools.turn_left(decision.turn_degrees or 15.0)

    if action is DroneAction.TURN_RIGHT:
        return await tools.turn_right(decision.turn_degrees or 15.0)

    if action is DroneAction.ADJUST_MOTOR:
        speed = decision.target_motor_speed
        if speed is None:
            # Model asked to adjust but gave no value — safest is to hold.
            return {"ok": False, "action": "adjust_motor", "error": "no target_motor_speed"}
        return await tools.adjust_motor_speed("all", speed)

    # HOLD (or anything unexpected): take no actuator action.
    return {"ok": True, "action": "hold", "motors": drone_state.motors.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# One decision (used by the live loop AND the benchmark)
# ─────────────────────────────────────────────────────────────────────────────

async def decide(
    agent: Agent[None, DroneDecision],
    telemetry: dict,
    frame_b64: Optional[str] = None,
    *,
    use_vision: bool = True,
    history=None,
) -> tuple[DroneDecision, float, bool]:
    """
    Run the agent for one tick and return (decision, latency_ms, used_vision).

    Raises on model failure — callers decide how to fall back.
    """
    import asyncio

    prompt, used_vision = build_prompt(telemetry, frame_b64, use_vision, history)
    with stopwatch() as t:
        # Hard ceiling on top of the HTTP-level ModelSettings timeout: even if the
        # model stalls while loading into VRAM, the control loop can never hang
        # indefinitely — it raises TimeoutError and the caller falls back safely.
        result = await asyncio.wait_for(
            agent.run(prompt),
            timeout=config.OLLAMA_TIMEOUT_S + 10.0,
        )
    return result.output, t.ms, used_vision


def _rule_action_for_tilt(tilt: float) -> tuple[DroneAction, HazardLevel]:
    """
    The deterministic rule-engine mapping from tilt angle to (action, hazard).
    Shared by the reflex layer and the safe fallback so the two can never diverge
    from the Phase-3 thresholds.
    """
    if tilt >= config.TILT_DANGER_DEG:
        return DroneAction.STABILIZE, HazardLevel.CRITICAL
    if tilt >= config.TILT_WARNING_DEG:
        return DroneAction.STABILIZE, HazardLevel.MEDIUM
    return DroneAction.HOVER, HazardLevel.NONE


def reflex_decision(telemetry: dict) -> DroneDecision:
    """
    The fast REFLEX layer's verdict: the Phase-3 IF/THEN rule engine expressed as
    a ``DroneDecision``. Deterministic, sub-millisecond, always fully confident.
    This is the safety floor of the hybrid controller.
    """
    tilt = compute_tilt(telemetry)
    action, hazard = _rule_action_for_tilt(tilt)
    if hazard is HazardLevel.CRITICAL:
        why = f"Reflex: tilt {tilt:.1f}° ≥ {config.TILT_DANGER_DEG:.0f}° danger → stabilize."
    elif hazard is HazardLevel.MEDIUM:
        why = f"Reflex: tilt {tilt:.1f}° ≥ {config.TILT_WARNING_DEG:.0f}° warning → stabilize."
    else:
        why = f"Reflex: tilt {tilt:.1f}° nominal → hover."
    return DroneDecision(action=action, hazard_level=hazard, reasoning=why, confidence=1.0)


def arbitrate(
    reflex: DroneDecision,
    deliberate: Optional[DroneDecision],
    tilt: float,
) -> tuple[DroneDecision, str]:
    """
    Merge the REFLEX (deterministic safety) and DELIBERATION (LLM context) layers
    into one decision — a subsumption arbiter where safety is non-negotiable.

    The reflex can always *veto* the agent; the agent can never relax safety below
    the reflex floor. Returns (chosen, source) where source explains the call:

      "reflex-override"  tilt ≥ DANGER → reflex acts at once; the agent is bypassed.
      "reflex-only"      no deliberation available (agent slow / unavailable).
      "reflex-floor"     reflex demands STABILIZE but the agent tried something
                         less safe → the reflex floor is enforced.
      "deliberation"     situation is safe → the agent's richer decision is used.
    """
    if tilt >= config.TILT_DANGER_DEG:
        return reflex, "reflex-override"
    if deliberate is None:
        return reflex, "reflex-only"
    if reflex.action is DroneAction.STABILIZE and deliberate.action not in (
        DroneAction.STABILIZE,
        DroneAction.EMERGENCY_STOP,
    ):
        return reflex, "reflex-floor"
    return deliberate, "deliberation"


def safe_fallback_decision(telemetry: dict, error: str) -> DroneDecision:
    """
    Deterministic safe action used when the LLM call fails entirely.
    Mirrors the rule engine so the drone never goes uncontrolled.
    """
    action, hazard = _rule_action_for_tilt(compute_tilt(telemetry))
    return DroneDecision(
        action=action,
        hazard_level=hazard,
        reasoning=f"[FALLBACK] LLM unavailable ({error}); applied rule-based safe action.",
        confidence=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Live server: one tick + the background loop
# ─────────────────────────────────────────────────────────────────────────────

def _log_and_record(
    *,
    log_source: str,
    mode: str,
    decision: DroneDecision,
    tilt: float,
    latency_ms: float,
    used_vision: bool,
    ok: bool,
    error: Optional[str],
    scenario: Optional[str],
    collector: Optional[MetricsCollector],
    source: Optional[str] = None,
) -> DecisionRecord:
    """
    Emit the explainability log line (dashboard) AND the DecisionRecord (metrics)
    for one control tick. Shared by the agentic and hybrid paths so both produce
    an identical schema. ``source`` (hybrid arbitration outcome) is woven into the
    message/record when present.
    """
    level = "INFO"
    if decision.hazard_level in (HazardLevel.HIGH, HazardLevel.CRITICAL):
        level = "ERROR"
    elif decision.hazard_level is HazardLevel.MEDIUM:
        level = "WARNING"
    if not ok:
        level = "ERROR"

    prefix = f"[{source}] " if source else ""
    extra = {
        "action": decision.action.value,
        "hazard": decision.hazard_level.value,
        "confidence": decision.confidence,
        "latency_ms": round(latency_ms, 1),
        "used_vision": used_vision,
        "ok": ok,
    }
    if source:
        extra["source"] = source

    drone_state.add_log(
        log_source,
        f"{prefix}{decision.action.value} ({decision.hazard_level.value}) "
        f"@ tilt {tilt:.1f}° — {decision.reasoning}",
        level=level,
        extra=extra,
    )

    rec = DecisionRecord(
        ts=__import_time(),
        mode=mode,
        action=decision.action.value,
        latency_ms=latency_ms,
        tilt_deg=tilt,
        hazard_level=decision.hazard_level.value,
        motor_speed=drone_state.motors.fl,
        confidence=decision.confidence,
        reasoning=f"{prefix}{decision.reasoning}",
        used_vision=used_vision,
        ok=ok,
        error=error,
        scenario=scenario,
    )
    (collector or metrics).record(rec)
    return rec


async def agent_step(
    agent: Optional[Agent[None, DroneDecision]] = None,
    *,
    scenario: Optional[str] = None,
    collector: Optional[MetricsCollector] = None,
) -> DecisionRecord:
    """
    Perform one agentic control tick against the *live* drone_state:
    read latest frame+telemetry → LLM → dispatch → log + record metrics.

    Always returns a DecisionRecord; never raises.
    """
    agent = agent or get_live_agent()
    telemetry = dict(drone_state.latest_telemetry or {})
    frame_b64 = drone_state.latest_frame_b64
    tilt = compute_tilt(telemetry)

    used_vision = False
    ok = True
    error: Optional[str] = None

    try:
        decision, latency_ms, used_vision = await decide(
            agent, telemetry, frame_b64, use_vision=True,
            history=drone_state.telemetry_history,
        )
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        decision = safe_fallback_decision(telemetry, error)
        latency_ms = 0.0
    except Exception as exc:  # network errors, Ollama down, timeouts, etc.
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        decision = safe_fallback_decision(telemetry, error)
        latency_ms = 0.0

    # Apply the decision through the MCP tool layer.
    try:
        await dispatch_decision(decision)
    except Exception as exc:
        ok = False
        error = (error + " | " if error else "") + f"dispatch: {exc}"

    # Explainability log (dashboard) + metrics record, via the shared emitter.
    return _log_and_record(
        log_source="AGENT",
        mode="agentic",
        decision=decision,
        tilt=tilt,
        latency_ms=latency_ms,
        used_vision=used_vision,
        ok=ok,
        error=error,
        scenario=scenario,
        collector=collector,
    )


def __import_time() -> float:
    import time

    return time.time()


async def _agent_step_shadow(*, active: bool) -> None:
    """
    Run one agent inference tick.

    Always stores result to ``drone_state.shadow_decisions["agent"]`` so the
    compare strip on the dashboard always has a fresh agent verdict.
    When ``active=True`` also dispatches motor commands and writes to the main
    decision log — identical to the old ``agent_step()`` behaviour.
    """
    import time as _time

    agent = get_live_agent()
    telemetry  = dict(drone_state.latest_telemetry or {})
    frame_b64  = drone_state.latest_frame_b64
    tilt       = compute_tilt(telemetry)
    used_vision = False
    ok         = True
    error: Optional[str] = None
    latency_ms = 0.0

    try:
        decision, latency_ms, used_vision = await decide(
            agent, telemetry, frame_b64, use_vision=True,
            history=drone_state.telemetry_history,
        )
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        ok, error = False, f"{type(exc).__name__}: {exc}"
        decision = safe_fallback_decision(telemetry, error)
    except Exception as exc:
        ok, error = False, f"{type(exc).__name__}: {exc}"
        decision = safe_fallback_decision(telemetry, error)

    drone_state.shadow_decisions["agent"] = {
        "action":      decision.action.value,
        "latency_ms":  round(latency_ms, 1),
        "tilt_deg":    tilt,
        "confidence":  decision.confidence,
        "hazard":      decision.hazard_level.value,
        "reasoning":   decision.reasoning,
        "used_vision": used_vision,
        "ok":          ok,
        "ts":          _time.time(),
    }

    if active:
        try:
            await dispatch_decision(decision)
        except Exception as exc2:
            ok = False
            error = (error + " | " if error else "") + f"dispatch: {exc2}"
        _log_and_record(
            log_source="AGENT",
            mode="agentic",
            decision=decision,
            tilt=tilt,
            latency_ms=latency_ms,
            used_vision=used_vision,
            ok=ok,
            error=error,
            scenario=None,
            collector=None,
        )


async def agentic_agent_loop() -> None:
    """
    Background task that drives agent inference continuously (not just when
    mode == 'agentic').  When inactive the result is shadow-only (no motor
    commands, no main log), so the compare strip always has a live agent verdict.
    """
    import asyncio

    drone_state.add_log("AGENT", "Agentic loop ready.")
    while True:
        await asyncio.sleep(config.AGENT_INFERENCE_INTERVAL)
        if not drone_state.drone_connected:
            continue
        try:
            await _agent_step_shadow(active=(drone_state.agent_mode == "agentic"))
        except Exception as exc:
            drone_state.add_log("AGENT", f"Agent loop error: {exc}", level="ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid controller: fast reflex floor + slow LLM deliberation
# ─────────────────────────────────────────────────────────────────────────────

async def hybrid_step(
    agent: Optional[Agent[None, DroneDecision]] = None,
    *,
    scenario: Optional[str] = None,
    collector: Optional[MetricsCollector] = None,
) -> DecisionRecord:
    """
    One hybrid control tick (used by the benchmark and the test-suite).

    Fast path: when the reflex layer sees DANGER it acts immediately and the LLM
    is skipped entirely — hybrid is then exactly as fast as the rule engine.
    Otherwise the agent deliberates and :func:`arbitrate` merges the two with the
    reflex as a safety floor. Mirrors :func:`agent_step` and never raises.

    The recorded ``latency_ms`` is the latency of the *applied* decision: 0 when
    the reflex handled the tick, the agent's latency only when its verdict was
    actually used. (In the live loop the agent runs in a separate coroutine, so
    its cost never blocks the reflex; see :func:`hybrid_controller_loop`.)
    """
    telemetry = dict(drone_state.latest_telemetry or {})
    frame_b64 = drone_state.latest_frame_b64
    tilt = compute_tilt(telemetry)
    reflex = reflex_decision(telemetry)

    used_vision = False
    ok = True
    error: Optional[str] = None
    latency_ms = 0.0

    if tilt >= config.TILT_DANGER_DEG:
        # Reflex owns control in danger — don't pay for the LLM at all.
        chosen, source = reflex, "reflex-override"
    else:
        agent = agent or get_live_agent()
        deliberate: Optional[DroneDecision] = None
        try:
            deliberate, latency_ms, used_vision = await decide(
                agent, telemetry, frame_b64, use_vision=True,
                history=drone_state.telemetry_history,
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            latency_ms = 0.0
        except Exception as exc:  # network errors, Ollama down, timeouts, etc.
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            latency_ms = 0.0
        chosen, source = arbitrate(reflex, deliberate, tilt)

    # Apply the chosen decision through the MCP tool layer.
    try:
        await dispatch_decision(chosen)
    except Exception as exc:
        ok = False
        error = (error + " | " if error else "") + f"dispatch: {exc}"

    applied = source == "deliberation"
    return _log_and_record(
        log_source="HYBRID",
        mode="hybrid",
        decision=chosen,
        tilt=tilt,
        latency_ms=latency_ms if applied else 0.0,
        used_vision=used_vision if applied else False,
        ok=ok,
        error=error,
        scenario=scenario,
        collector=collector,
        source=source,
    )


async def _hybrid_reflex_tick(last_key: Optional[tuple]) -> Optional[tuple]:
    """
    One fast reflex tick for the live loop: arbitrate the reflex against the
    cached deliberation, apply the result, and — only when the applied behaviour
    changes — log + record it (the reflex fires at 10 Hz; we must not flood).
    """
    import time

    telemetry = dict(drone_state.latest_telemetry or {})
    tilt = compute_tilt(telemetry)
    reflex = reflex_decision(telemetry)

    deliberate: Optional[DroneDecision] = None
    delib_latency = 0.0
    delib_vision = False
    cache = drone_state.hybrid_deliberation
    if (
        tilt < config.TILT_DANGER_DEG
        and cache
        and (time.time() - float(cache.get("ts", 0.0))) <= config.HYBRID_DELIBERATION_TTL_S
    ):
        try:
            deliberate = DroneDecision(
                action=DroneAction(cache["action"]),
                hazard_level=HazardLevel(cache["hazard"]),
                reasoning=cache.get("reasoning", ""),
                confidence=float(cache.get("confidence", 0.7) or 0.7),
                target_motor_speed=cache.get("target_motor_speed"),
                turn_degrees=cache.get("turn_degrees"),
            )
            delib_latency = float(cache.get("latency_ms", 0.0) or 0.0)
            delib_vision = bool(cache.get("used_vision", False))
        except Exception:
            deliberate = None

    chosen, source = arbitrate(reflex, deliberate, tilt)
    await dispatch_decision(chosen)

    key = (chosen.action.value, source)
    if key != last_key:
        applied = source == "deliberation"
        _log_and_record(
            log_source="HYBRID",
            mode="hybrid",
            decision=chosen,
            tilt=tilt,
            latency_ms=delib_latency if applied else 0.0,
            used_vision=delib_vision if applied else False,
            ok=True,
            error=None,
            scenario=None,
            collector=None,
            source=source,
        )
    return key


async def hybrid_controller_loop() -> None:
    """
    Background task (registered in main.py) for the LIVE hybrid controller.

    Two asyncio cadences share one cache on ``drone_state``:
      * REFLEX       — fast (HYBRID_REFLEX_HZ): deterministic safety EVERY tick.
      * DELIBERATION — slow (AGENT_INFERENCE_HZ): the LLM refines behaviour when
        the situation is safe; its verdict is cached for the reflex to consume.

    Because the agent runs in its own coroutine, a slow (~seconds) inference never
    delays the reflex — the drone keeps being stabilised at 10 Hz while the agent
    is still thinking. If Ollama is down the cache simply never refreshes and the
    reflex carries the whole load: graceful degradation, never a crash.
    """
    import asyncio
    import time

    drone_state.add_log("HYBRID", "Hybrid loop ready (reflex + deliberation).")

    async def _deliberation_worker() -> None:
        agent: Optional[Agent[None, DroneDecision]] = None
        last_fail_log = 0.0
        while True:
            await asyncio.sleep(config.AGENT_INFERENCE_INTERVAL)
            if drone_state.agent_mode != "hybrid" or not drone_state.drone_connected:
                continue
            telemetry = dict(drone_state.latest_telemetry or {})
            # In danger the reflex owns control; don't spend inference time.
            if not telemetry or compute_tilt(telemetry) >= config.TILT_DANGER_DEG:
                continue
            if agent is None:
                agent = get_live_agent()
            try:
                deliberate, latency_ms, used_vision = await decide(
                    agent, telemetry, drone_state.latest_frame_b64,
                    use_vision=True, history=drone_state.telemetry_history,
                )
                drone_state.hybrid_deliberation = {
                    "action": deliberate.action.value,
                    "hazard": deliberate.hazard_level.value,
                    "reasoning": deliberate.reasoning,
                    "confidence": deliberate.confidence,
                    "target_motor_speed": deliberate.target_motor_speed,
                    "turn_degrees": deliberate.turn_degrees,
                    "latency_ms": latency_ms,
                    "used_vision": used_vision,
                    "ts": time.time(),
                }
            except Exception as exc:
                now = time.time()
                if now - last_fail_log > 10.0:
                    drone_state.add_log(
                        "HYBRID",
                        f"Deliberation unavailable ({exc}); reflex maintaining safety.",
                        level="WARNING",
                    )
                    last_fail_log = now

    worker = asyncio.create_task(_deliberation_worker(), name="hybrid-deliberation")
    last_key: Optional[tuple] = None
    try:
        while True:
            await asyncio.sleep(config.HYBRID_REFLEX_INTERVAL)
            if drone_state.agent_mode != "hybrid" or not drone_state.drone_connected:
                continue
            try:
                last_key = await _hybrid_reflex_tick(last_key)
            except Exception as exc:  # final safety net — the reflex must never die
                drone_state.add_log("HYBRID", f"Reflex tick error: {exc}", level="ERROR")
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Ollama health check / warm-up
# ─────────────────────────────────────────────────────────────────────────────

async def check_ollama() -> dict:
    """Probe the Ollama server: is it up, and is the configured model present?"""
    import httpx

    info: dict = {"reachable": False, "model_available": False, "models": []}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            info["reachable"] = True
            info["models"] = names
            base = config.OLLAMA_MODEL.split(":")[0]
            info["model_available"] = any(
                n == config.OLLAMA_MODEL or n.split(":")[0] == base for n in names
            )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# Convenience alias matching the type used in benchmark/tests.
DecisionFn = Callable[..., Awaitable[tuple[DroneDecision, float, bool]]]
