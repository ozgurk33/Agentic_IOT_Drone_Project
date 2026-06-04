"""
Navigation Agent — obstacle detection and directional commands.
Completely separate from the stabilization agent (agent_core.py).

The rule engine is BLIND: it reacts only to tilt angles.
This agent SEES: it analyzes camera frames for obstacles and suggests directions.

Demo value: point the phone camera at a wall while tilt=0 degrees.
  - Rule engine: "HOVER — tilt 0 degrees, all safe"  <- WRONG (doesn't see the wall)
  - Nav agent:   "Wall ahead -> TURN_LEFT"             <- CORRECT (camera used)
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.settings import ModelSettings

from backend import config
from backend.agent_brain.agent_core import _decode_frame, build_ollama_model
from backend.metrics import stopwatch
from backend.state import drone_state

logger = logging.getLogger("drone.nav")


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV zone analysis — brightness + coherence approach
# ─────────────────────────────────────────────────────────────────────────────

def _zone_analysis(frame_b64: str) -> tuple[str, str]:
    """
    Three-signal obstacle detector using brightness + color coherence analysis.

    Signal 1 — Zone boundary jump:
        At zone borders (left|center and center|right), measure the
        mean brightness difference over a thin vertical strip.
        Large jump on BOTH sides -> object FILLS center zone (close obstacle).
        Small jump -> same scene continues (open path).

    Signal 2 — Center vs side contrast:
        Mean brightness of center zone vs average of side zones.
        A close object creates a distinct brightness island.

    Signal 3 — Color coherence:
        Within center zone, fraction of pixels within 60 color units of
        the zone mean. High coherence (>60%) = one dominant surface close.

    This approach works for real JPEG camera frames which have natural
    brightness transitions at object boundaries. No Canny required.
    """
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return "", ""

        img  = cv2.resize(img, (300, 225))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bgr  = img.astype(np.float32)

        h, w = gray.shape
        y1, y2 = int(h * 0.18), int(h * 0.82)
        roi_g = gray[y1:y2, :]
        roi_b = bgr[y1:y2, :]

        zw = w // 3
        bw = max(zw // 6, 4)   # boundary strip width

        # Zone mean brightness
        bm = float(roi_g[:, :zw].mean())
        cm = float(roi_g[:, zw:2*zw].mean())
        rm = float(roi_g[:, 2*zw:].mean())

        # Signal 1: brightness jump at zone boundaries
        lbnd = float(roi_g[:, zw-bw:zw+bw].mean())
        rbnd = float(roi_g[:, 2*zw-bw:2*zw+bw].mean())
        b_left  = abs(cm - lbnd)
        b_right = abs(cm - rbnd)
        boundary_score = min(b_left, b_right)

        # Signal 2: center vs sides contrast
        side_mean = (bm + rm) / 2.0
        contrast  = abs(cm - side_mean)

        # Signal 3: color coherence in center zone
        cz = roi_b[:, zw:2*zw].reshape(-1, 3)
        cmean_col = cz.mean(axis=0)
        dists = np.abs(cz - cmean_col).sum(axis=1)
        coherence = float((dists < 60).mean())

        def classify_center() -> str:
            # Strong: clear boundary + contrast + coherent color
            if boundary_score > 22 and contrast > 18 and coherence > 0.55:
                return "ENGEL"
            # Strong: very large boundary alone
            if boundary_score > 38 and contrast > 12:
                return "ENGEL"
            # Strong: high coherence + contrast (uniform close surface)
            if coherence > 0.78 and contrast > 20:
                return "ENGEL"
            # Moderate
            if boundary_score > 15 and contrast > 13:
                return "MUHTEMEL"
            if coherence > 0.68 and contrast > 15:
                return "MUHTEMEL"
            return "ACIK"

        def classify_side(zone_m: float, other_m: float) -> str:
            diff_from_center = abs(zone_m - cm)
            diff_from_other  = abs(zone_m - other_m)
            if diff_from_center > 32 and diff_from_other > 22:
                return "ENGEL"
            if diff_from_center > 20 and diff_from_other > 14:
                return "MUHTEMEL"
            return "ACIK"

        cl = classify_center()
        ll = classify_side(bm, rm)
        rl = classify_side(rm, bm)

        hint = (
            f"ENGEL_ANALIZI: "
            f"SOL={ll}(br:{bm:.0f}) "
            f"MERKEZ={cl}(br:{cm:.0f},sinir:{boundary_score:.0f},coh:{coherence:.0%}) "
            f"SAG={rl}(br:{rm:.0f})"
        )

        fast = ""
        if cl == "ENGEL":
            # Only turn if ONE side is clearly blocked and the OTHER is clear
            if ll != "ACIK" and rl == "ACIK":
                fast = "TURN_RIGHT"   # left blocked → go right
            elif rl != "ACIK" and ll == "ACIK":
                fast = "TURN_LEFT"    # right blocked → go left
            else:
                fast = "STOP_OBSTACLE"  # both sides similar → stop
        elif ll == "ENGEL" and cl == "ACIK":
            fast = "TURN_RIGHT"
        elif rl == "ENGEL" and cl == "ACIK":
            fast = "TURN_LEFT"
        elif ll == "ACIK" and cl == "ACIK" and rl == "ACIK":
            fast = "MOVE_FORWARD"

        logger.debug("cv3: %s  fast=%s", hint, fast or "LLM")
        return hint, fast

    except Exception as exc:
        logger.debug("zone_analysis failed: %s", exc)
        return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

class NavDecision(BaseModel):
    action: str = Field(
        description=(
            "Navigation action: MOVE_FORWARD | TURN_LEFT | TURN_RIGHT | "
            "STOP_OBSTACLE | ASCEND | DESCEND | UNKNOWN"
        )
    )
    turn_degrees: Optional[float] = Field(
        default=None, ge=0, le=180,
        description="Degrees to turn (only for TURN_LEFT / TURN_RIGHT)",
    )
    obstacle_detected: bool = Field(
        default=False,
        description="True if any obstacle is visible in the camera frame",
    )
    obstacle_position: str = Field(
        default="none",
        description="Where the obstacle is: left | right | center | above | below | none",
    )
    what_i_see: str = Field(
        description="Max 3 words describing the scene (e.g. 'wall ahead', 'open room')",
    )
    reasoning: str = Field(
        description="Max 8 words explaining the decision",
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_NAV_PROMPT = """\
Drone kamerasi onundeki yolu kontrol et. Sadece JSON dondur.

Sana ENGEL_ANALIZI verisi gelecek (SOL/MERKEZ/SAG bolgeler icin parlaklik ve renk tutarlilik analizi):
  ENGEL   = o bolgede buyuk, yakin bir nesne var (yuksek sinir atlama + renk tutarli)
  MUHTEMEL = belirsiz
  ACIK    = yol acik

KARAR KURALLARI:
  MERKEZ=ENGEL                    -> STOP_OBSTACLE, obstacle_detected=true
  MERKEZ=ENGEL, SOL=ACIK          -> TURN_LEFT, obstacle_detected=true
  MERKEZ=ENGEL, SAG=ACIK          -> TURN_RIGHT, obstacle_detected=true
  SOL=ENGEL, MERKEZ=ACIK          -> TURN_RIGHT, obstacle_detected=true
  SAG=ENGEL, MERKEZ=ACIK          -> TURN_LEFT, obstacle_detected=true
  Hepsi ACIK veya MUHTEMEL        -> MOVE_FORWARD, obstacle_detected=false

what_i_see: MAX 3 KELIME (ornek: "duvar onde", "acik yol", "engel sol")
reasoning:  MAX 8 KELIME

JSON formatinda yanit ver."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent singleton
# ─────────────────────────────────────────────────────────────────────────────

_nav_agent: Optional[Agent[None, NavDecision]] = None


def reset_nav_agent() -> None:
    global _nav_agent
    _nav_agent = None


def get_nav_agent() -> Agent[None, NavDecision]:
    global _nav_agent
    if _nav_agent is None:
        _nav_agent = Agent(
            build_ollama_model(),
            output_type=NativeOutput(NavDecision),
            instructions=_NAV_PROMPT,
            retries=1,
            model_settings=ModelSettings(
                temperature=0.1,
                timeout=config.OLLAMA_TIMEOUT_S,
            ),
            name="drone-nav-controller",
        )
    return _nav_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def nav_analyze(frame_b64: Optional[str] = None) -> dict:
    """
    Analyze camera frame for obstacles.

    Pipeline:
      1. OpenCV brightness/coherence analysis -> zone labels + optional fast path
      2. If fast path unambiguous -> skip LLM, return immediately (< 30ms)
      3. Otherwise -> LLM with zone hint + image for final decision
    """
    if frame_b64 is None:
        frame_b64 = drone_state.latest_frame_b64

    if not frame_b64:
        result = {
            "action": "UNKNOWN",
            "turn_degrees": None,
            "obstacle_detected": False,
            "obstacle_position": "none",
            "what_i_see": "Kamera goruntusu yok",
            "reasoning": "Kamera bagli degil.",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "used_vision": False,
            "ok": False,
            "error": "no_frame",
            "rule_would_say": _rule_blind_verdict(),
        }
        drone_state.shadow_decisions["nav"] = result
        return result

    # Step 1: OpenCV zone analysis
    zone_hint, fast_action = _zone_analysis(frame_b64)

    # Step 2: Fast path (no LLM needed)
    if fast_action:
        pos_map = {
            "STOP_OBSTACLE": "center",
            "TURN_LEFT":     "right",
            "TURN_RIGHT":    "left",
        }
        result = {
            "action": fast_action,
            "turn_degrees": 30 if fast_action in ("TURN_LEFT", "TURN_RIGHT") else None,
            "obstacle_detected": fast_action != "MOVE_FORWARD",
            "obstacle_position": pos_map.get(fast_action, "none"),
            "what_i_see": zone_hint.split("MERKEZ=")[-1][:20].strip(),
            "reasoning": f"Goruntu analizi: {zone_hint[:60]}",
            "confidence": 0.88,
            "latency_ms": 5.0,
            "used_vision": True,
            "ok": True,
            "rule_would_say": _rule_blind_verdict(),
            "fast_path": True,
        }
        drone_state.shadow_decisions["nav"] = result
        _log_nav(result)
        return result

    # Step 3: LLM with zone hint + image
    agent = get_nav_agent()
    jpeg = _decode_frame(frame_b64)
    used_vision = False
    latency_ms = 0.0

    prompt_text = f"{zone_hint}\n\nKamera karesini analiz et:" if zone_hint else "Kamera karesini analiz et:"
    parts: list = [prompt_text]
    if jpeg:
        parts.append(BinaryContent(data=jpeg, media_type="image/jpeg"))
        used_vision = True

    try:
        with stopwatch() as t:
            out = await agent.run(parts)
        dec: NavDecision = out.output
        latency_ms = t.ms

        result = {
            "action": dec.action,
            "turn_degrees": dec.turn_degrees,
            "obstacle_detected": dec.obstacle_detected,
            "obstacle_position": dec.obstacle_position,
            "what_i_see": dec.what_i_see,
            "reasoning": dec.reasoning,
            "confidence": dec.confidence,
            "latency_ms": round(latency_ms, 1),
            "used_vision": used_vision,
            "ok": True,
            "rule_would_say": _rule_blind_verdict(),
            "zone_hint": zone_hint,
        }

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("Nav agent error: %s", err)
        result = {
            "action": "UNKNOWN",
            "turn_degrees": None,
            "obstacle_detected": False,
            "obstacle_position": "none",
            "what_i_see": "Model analiz yapamadi",
            "reasoning": f"Hata: {err}",
            "confidence": 0.0,
            "latency_ms": round(latency_ms, 1),
            "used_vision": used_vision,
            "ok": False,
            "error": err,
            "rule_would_say": _rule_blind_verdict(),
        }

    drone_state.shadow_decisions["nav"] = result
    _log_nav(result)
    return result


def _log_nav(result: dict) -> None:
    level = "WARNING" if result.get("obstacle_detected") else "INFO"
    action = result.get("action", "?")
    see = result.get("what_i_see", "")[:50]
    drone_state.add_log("NAV", f"[{action}] {see}", level=level, extra={**result, "source": "NAV"})


def _rule_blind_verdict() -> str:
    tele = drone_state.latest_telemetry or {}
    beta  = abs(float(tele.get("beta",  0.0) or 0.0))
    gamma = abs(float(tele.get("gamma", 0.0) or 0.0))
    tilt  = max(beta, gamma)
    if tilt >= config.TILT_DANGER_DEG:
        return f"STABILIZE (tilt {tilt:.1f}° >= {config.TILT_DANGER_DEG:.0f}°)"
    if tilt >= config.TILT_WARNING_DEG:
        return f"STABILIZE (tilt {tilt:.1f}° >= {config.TILT_WARNING_DEG:.0f}°)"
    return f"HOVER (tilt {tilt:.1f}° < {config.TILT_WARNING_DEG:.0f}° - KAMERA GORMUYOR)"
