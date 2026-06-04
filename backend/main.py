"""
Drone MCP Project — FastAPI + WebSocket Hub
===========================================
Serves:
  /              → Dashboard UI (frontend/dashboard/index.html)
  /drone         → Mobile client UI (frontend/drone_client/index.html)
  /ws/drone      → WebSocket: phone → server (frames + telemetry)
  /ws/dashboard  → WebSocket: server → browser (state broadcast)
  /mcp           → FastMCP SSE server (MCP protocol for Phase-5 agent)
  /health        → JSON health check
  /api/*         → REST control endpoints

Run:
    uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.metrics import DecisionRecord, metrics, stopwatch
from backend.state import drone_state          # ← shared singleton

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drone.server")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Connection Manager
# ─────────────────────────────────────────────────────────────────────────────

class DashboardManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        drone_state.dashboards_connected = len(self._clients)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        drone_state.dashboards_connected = len(self._clients)

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        message = json.dumps(payload, default=str)
        dead: Set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._clients.discard(ws)
        drone_state.dashboards_connected = len(self._clients)


dashboard_manager = DashboardManager()


# ─────────────────────────────────────────────────────────────────────────────
# Background Tasks
# ─────────────────────────────────────────────────────────────────────────────

async def broadcast_loop() -> None:
    """Push full state snapshot to all dashboards at BROADCAST_HZ."""
    while True:
        try:
            if drone_state.dashboards_connected > 0:
                payload = {
                    "type": "state_update",
                    "frame": drone_state.latest_frame_b64,
                    "telemetry": drone_state.latest_telemetry,
                    "mode": drone_state.agent_mode,
                    "drone_connected": drone_state.drone_connected,
                    "motors": drone_state.motors.to_dict(),
                    "logs": drone_state.decision_log[-30:],
                    "deliberation": drone_state.hybrid_deliberation,
                    "shadow_decisions": drone_state.shadow_decisions,
                }
                await dashboard_manager.broadcast(payload)
        except Exception as exc:
            logger.error("broadcast_loop: %s", exc)
        await asyncio.sleep(config.BROADCAST_INTERVAL)


# The three controllers the operator can switch between at runtime.
VALID_MODES = ("traditional", "agentic", "hybrid")


# Canonical action names so traditional & agentic metrics line up.
_RULE_TO_ACTION = {
    "NO_DATA": "HOLD",
    "EMERGENCY_STABILIZE": "STABILIZE",
    "STABILIZE": "STABILIZE",
    "HOVER": "HOVER",
}


_RULE_TO_HAZARD = {
    "EMERGENCY_STABILIZE": "CRITICAL",
    "STABILIZE": "MEDIUM",
    "HOVER": "NONE",
    "NO_DATA": "NONE",
}


async def traditional_rule_loop() -> None:
    """
    Run the IF/THEN rule engine continuously (always, not just when active).

    When mode == 'traditional': applies motor commands + writes to main log.
    Otherwise: runs the pure compute path (no side effects) to populate the
    shadow_decisions store, so the compare strip always has a fresh rule verdict.
    """
    from backend.mcp_server.mcp_server import (
        run_traditional_rules,
        compute_rule_decision_only,
    )

    while True:
        await asyncio.sleep(config.AGENT_INFERENCE_INTERVAL)
        if not drone_state.drone_connected:
            continue
        is_active = drone_state.agent_mode == "traditional"
        try:
            with stopwatch() as t:
                if is_active:
                    result = await run_traditional_rules()
                else:
                    result = compute_rule_decision_only()

            rule     = result.get("rule", "HOVER")
            action   = _RULE_TO_ACTION.get(rule, rule)
            tilt_deg = result.get("tilt_deg")
            hazard   = _RULE_TO_HAZARD.get(rule, "NONE")

            drone_state.shadow_decisions["rule"] = {
                "action": action,
                "latency_ms": round(t.ms, 2),
                "tilt_deg": tilt_deg,
                "confidence": 1.0,
                "hazard": hazard,
                "reasoning": result.get("reasoning", f"rule:{rule}"),
                "ok": True,
                "ts": time.time(),
            }

            if is_active:
                metrics.record(DecisionRecord(
                    ts=time.time(),
                    mode="traditional",
                    action=action,
                    latency_ms=t.ms,
                    tilt_deg=tilt_deg,
                    hazard_level=hazard,
                    motor_speed=result.get("motor_speed"),
                    confidence=1.0,
                    reasoning=f"rule:{rule}",
                    used_vision=False,
                    ok=True,
                ))
        except Exception as exc:
            drone_state.add_log("RULE", f"Rule engine error: {exc}", level="ERROR")


async def agentic_loop() -> None:
    """Drive the Pydantic-AI agent when mode='agentic' (lazy import)."""
    from backend.agent_brain.agent_core import agentic_agent_loop

    await agentic_agent_loop()


async def hybrid_loop() -> None:
    """Drive the hybrid reflex+deliberation controller when mode='hybrid'."""
    from backend.agent_brain.agent_core import hybrid_controller_loop

    await hybrid_controller_loop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 62)
    logger.info("  Drone MCP Project — Rule-Based + Agentic + Hybrid")
    logger.info("  Dashboard  : http://<your-ip>:%d/", config.PORT)
    logger.info("  Drone URL  : http://<your-ip>:%d/drone", config.PORT)
    logger.info("  MCP SSE    : http://localhost:%d/mcp/sse", config.PORT)
    logger.info("  Health     : http://localhost:%d/health", config.PORT)
    logger.info("  Ollama     : %s  (model: %s)", config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    logger.info("=" * 62)

    # One-off Ollama probe so the operator immediately knows if agentic mode
    # will work — never blocks startup.
    try:
        from backend.agent_brain.agent_core import check_ollama
        probe = await check_ollama()
        if probe.get("reachable"):
            ok = "✓" if probe.get("model_available") else "✗ (model not pulled!)"
            logger.info("  Ollama reachable ✓  | model '%s' %s", config.OLLAMA_MODEL, ok)
        else:
            logger.warning("  Ollama NOT reachable — agentic mode will use safe fallback.")
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("  Ollama probe failed: %s", exc)

    tasks = [
        asyncio.create_task(broadcast_loop(),        name="broadcast"),
        asyncio.create_task(traditional_rule_loop(), name="rule-engine"),
        asyncio.create_task(agentic_loop(),          name="agent"),
        asyncio.create_task(hybrid_loop(),           name="hybrid"),
    ]
    yield
    for t in tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("Server shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drone MCP Server",
    description="Smartphone ↔ AI Brain — Drone Simulation",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static assets ─────────────────────────────────────────────────────────────
if config.FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")

# ── Mount FastMCP as SSE sub-app at /mcp ─────────────────────────────────────
# This gives the Phase-5 Pydantic-AI agent an SSE endpoint at:
#   http://localhost:8000/mcp/sse
from backend.mcp_server.mcp_server import mcp as _mcp_app   # noqa: E402

app.mount("/mcp", _mcp_app.http_app(transport="sse"))


# ─────────────────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "mode": drone_state.agent_mode,
        "drone_connected": drone_state.drone_connected,
        "dashboards_connected": drone_state.dashboards_connected,
        "motors": drone_state.motors.to_dict(),
        "log_count": len(drone_state.decision_log),
    })


@app.post("/api/mode/{mode}", tags=["control"])
async def set_mode(mode: str) -> JSONResponse:
    if mode not in VALID_MODES:
        return JSONResponse(
            {"error": f"mode must be one of {VALID_MODES}"}, status_code=400
        )
    drone_state.agent_mode = mode
    drone_state.add_log("API", f"Mode → {mode.upper()}")
    await dashboard_manager.broadcast({"type": "mode_change", "mode": mode})
    return JSONResponse({"mode": mode, "ok": True})


@app.get("/api/logs", tags=["meta"])
async def get_logs(limit: int = 100) -> JSONResponse:
    return JSONResponse({"logs": drone_state.decision_log[-limit:]})


@app.get("/api/telemetry", tags=["meta"])
async def get_telemetry() -> JSONResponse:
    return JSONResponse(drone_state.latest_telemetry)


@app.get("/api/motors", tags=["meta"])
async def get_motors() -> JSONResponse:
    return JSONResponse(drone_state.motors.to_dict())


@app.get("/api/metrics", tags=["meta"])
async def get_metrics(limit: int = 100) -> JSONResponse:
    """Recent per-decision metric records (both modes)."""
    return JSONResponse({"records": metrics.recent(limit)})


@app.get("/api/metrics/summary", tags=["meta"])
async def get_metrics_summary() -> JSONResponse:
    """Aggregate latency / action mix split by mode — drives the comparison view."""
    return JSONResponse(metrics.summary())


@app.get("/api/ollama", tags=["meta"])
async def get_ollama_status() -> JSONResponse:
    """Live probe of the Ollama server and configured model."""
    from backend.agent_brain.agent_core import check_ollama

    info = await check_ollama()
    info["configured_model"] = config.OLLAMA_MODEL
    info["base_url"] = config.OLLAMA_BASE_URL
    return JSONResponse(info)


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenario injection  (no phone needed — injects synthetic telemetry)
# ─────────────────────────────────────────────────────────────────────────────

_demo_task: Optional[asyncio.Task] = None


async def _inject_fake_telemetry(beta: float, gamma: float, alpha: float = 0.0) -> None:
    t: dict = {
        "type": "telemetry",
        "ax": 0.0, "ay": 0.0, "az": 9.8,
        "alpha": round(alpha, 2),
        "beta": round(beta, 2),
        "gamma": round(gamma, 2),
        "tilt": round(max(abs(beta), abs(gamma)), 2),
        "ts": time.time(),
    }
    drone_state.note_telemetry(t)


async def _scenario_calm() -> None:
    import math
    tick = 0.0
    while True:
        tick += 0.08
        gamma = math.sin(tick * 0.5) * 3.0   # roll (left-right) — visible from front
        beta  = math.sin(tick * 0.7) * 2.0   # small pitch noise
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)


async def _scenario_gradual_tilt() -> None:
    import math
    i = 0
    while True:
        gamma = min(i * 0.45, 45.0)           # roll (left-right) — clearly visible
        if i > 80:
            gamma = max(45.0 - (i - 80) * 0.6, 0.0)
        beta = math.sin(i * 0.18) * 4.0       # small pitch noise
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)
        i = (i + 1) % 160


async def _scenario_shake() -> None:
    import math
    import random
    i = 0
    while True:
        await _inject_fake_telemetry(
            math.sin(i * 0.7) * 38.0 + random.uniform(-4, 4),
            math.cos(i * 1.0) * 32.0,
        )
        await asyncio.sleep(0.08)
        i += 1


async def _scenario_critical() -> None:
    """0° → rapid 43° emergency → recovery → 0°. Clear HOVER→EMERGENCY→HOVER cycle."""
    import math
    i = 0
    while True:
        phase = i % 130
        if phase < 8:                          # 0.8s calm at 0°
            beta, gamma = 0.0, 0.0
        elif phase < 35:                       # rapid rise 0→43°
            frac = (phase - 8) / 27.0
            gamma = 43.0 * frac               # roll (visible left-right tilt)
            beta  = 5.0 * frac
        elif phase < 65:                       # hold at 43° — EMERGENCY
            gamma = 43.0
            beta  = math.sin(phase * 0.3) * 4.0
        elif phase < 100:                      # recovery 43→0°
            frac  = (phase - 65) / 35.0
            gamma = 43.0 * (1.0 - frac)
            beta  = 4.0 * (1.0 - frac)
        else:                                  # 3s calm at 0°
            beta, gamma = 0.0, 0.0
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)
        i += 1


async def _scenario_turbulence() -> None:
    """High-frequency random jitter — simulates flying in gusty wind."""
    import math
    import random
    t = 0.0
    while True:
        t += 0.1
        beta  = math.sin(t * 2.1) * 18 + random.uniform(-8, 8)
        gamma = math.cos(t * 1.7) * 15 + random.uniform(-6, 6)
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.08)


async def _scenario_wind_gust() -> None:
    """Calm → sudden 40° gust → full recovery to 0° — tests emergency response speed."""
    import math as _math
    i = 0
    while True:
        phase = i % 90
        if phase < 20:
            gamma = phase * 2.0         # 0→40° roll (visible left-right)
        elif phase < 35:
            gamma = 40.0                # peak danger
        elif phase < 60:
            gamma = 40.0 - (phase - 35) * 1.6   # recovery 40→0°
        else:
            gamma = 0.0                 # calm
        beta = _math.sin(i * 0.3) * 3 if phase > 60 else _math.sin(i * 0.3) * 4
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)
        i += 1


async def _scenario_zigzag() -> None:
    """Left-right banking turns — tests TURN_LEFT / TURN_RIGHT decisions."""
    import math
    t = 0.0
    while True:
        t += 0.06
        gamma = math.sin(t * 0.8) * 22.0    # roll left/right
        beta  = math.cos(t * 1.2) * 8.0     # slight pitch
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)


async def _scenario_obstacle() -> None:
    """
    OBSTACLE DEMO — the killer 'rule is blind, AI sees' scenario.
    Forces telemetry to flat (0° tilt) so the rule engine always says HOVER.
    The camera feed (real phone or test) shows the obstacle the AI must detect.
    """
    while True:
        await _inject_fake_telemetry(0.0, 0.0)   # perfectly level — rule says HOVER
        await asyncio.sleep(0.1)


async def _scenario_ai_advantage() -> None:
    """
    AI SUPERIORITY demo — tilt stays BELOW the 15° rule threshold the whole time,
    so the rule engine always says HOVER.  But tilt is steadily RISING (0→13°),
    which the agent detects via TILT_TREND and responds with STABILIZE.

    Rule: HOVER throughout  <- can't see the trend
    Agent: STABILIZE when trend shows danger <- contextual reasoning
    """
    import math
    i = 0
    while True:
        phase = i % 120
        if phase < 40:
            gamma = phase * 0.325          # 0→13° roll — visible, below 15° threshold
        elif phase < 70:
            gamma = 13.0                   # holds at 13° — rule still says HOVER
        else:
            gamma = max(0.0, 13.0 - (phase - 70) * 0.43)
        beta = math.sin(i * 0.25) * 2.0   # small pitch noise
        await _inject_fake_telemetry(beta, gamma)
        await asyncio.sleep(0.1)
        i += 1


_SCENARIOS: dict = {
    "calm":         _scenario_calm,
    "gradual_tilt": _scenario_gradual_tilt,
    "shake":        _scenario_shake,
    "critical":     _scenario_critical,
    "turbulence":   _scenario_turbulence,
    "wind_gust":    _scenario_wind_gust,
    "zigzag":       _scenario_zigzag,
    "ai_advantage": _scenario_ai_advantage,
    "obstacle":     _scenario_obstacle,
}


@app.post("/api/demo/scenario/{name}", tags=["demo"])
async def start_demo_scenario(name: str) -> JSONResponse:
    """Start a synthetic flight scenario — no phone needed."""
    global _demo_task
    if name not in _SCENARIOS:
        return JSONResponse({"error": f"Unknown scenario. Valid: {list(_SCENARIOS)}"}, status_code=400)
    if _demo_task and not _demo_task.done():
        _demo_task.cancel()
    # Clear stale telemetry so old values don't flash on scenario switch
    drone_state.latest_telemetry = None
    drone_state.shadow_decisions["rule"]  = None
    drone_state.shadow_decisions["agent"] = None
    drone_state.drone_connected = True
    _demo_task = asyncio.create_task(_SCENARIOS[name]())
    drone_state.add_log("DEMO", f"Senaryo başladı: {name.upper()}")
    await dashboard_manager.broadcast({"type": "drone_status", "connected": True})
    return JSONResponse({"ok": True, "scenario": name})


# ─────────────────────────────────────────────────────────────────────────────
# Live threshold editor  (jüri demosunda "bak şu an değiştiriyorum" efekti)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/config/thresholds", tags=["config"])
async def get_thresholds() -> JSONResponse:
    return JSONResponse({
        "tilt_warning":     config.TILT_WARNING_DEG,
        "tilt_danger":      config.TILT_DANGER_DEG,
        "motor_default":    config.MOTOR_DEFAULT,
        "motor_stabilize":  config.MOTOR_STABILIZE,
        "motor_emergency":  config.MOTOR_EMERGENCY,
    })


@app.post("/api/config/thresholds", tags=["config"])
async def set_thresholds(body: dict) -> JSONResponse:
    """Update rule thresholds at runtime — agent rebuilds with new system prompt."""
    from backend.agent_brain.agent_core import switch_model

    changed: list[str] = []
    if "tilt_warning" in body:
        config.TILT_WARNING_DEG = float(body["tilt_warning"])
        changed.append(f"UYARI={config.TILT_WARNING_DEG:.0f}°")
    if "tilt_danger" in body:
        config.TILT_DANGER_DEG  = float(body["tilt_danger"])
        changed.append(f"TEHLİKE={config.TILT_DANGER_DEG:.0f}°")
    if "motor_stabilize" in body:
        config.MOTOR_STABILIZE  = int(body["motor_stabilize"])
        changed.append(f"MOT_STAB={config.MOTOR_STABILIZE}%")
    if "motor_emergency" in body:
        config.MOTOR_EMERGENCY  = int(body["motor_emergency"])
        changed.append(f"MOT_EMRG={config.MOTOR_EMERGENCY}%")
    if "motor_default" in body:
        config.MOTOR_DEFAULT    = int(body["motor_default"])
        changed.append(f"MOT_DEF={config.MOTOR_DEFAULT}%")

    # Rebuild agent so its system prompt reflects the new thresholds
    switch_model(config.OLLAMA_MODEL)

    msg = "Eşikler güncellendi: " + ", ".join(changed)
    drone_state.add_log("CFG", msg)
    await dashboard_manager.broadcast({
        "type":          "config_update",
        "tilt_warning":  config.TILT_WARNING_DEG,
        "tilt_danger":   config.TILT_DANGER_DEG,
        "motor_default": config.MOTOR_DEFAULT,
        "motor_stabilize": config.MOTOR_STABILIZE,
        "motor_emergency": config.MOTOR_EMERGENCY,
    })
    return JSONResponse({"ok": True, "changed": changed})


# ─────────────────────────────────────────────────────────────────────────────
# Session CSV export
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/session/export.csv", tags=["meta"])
async def export_session_csv() -> object:
    """Download the current session's decision records as a CSV file."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    records = metrics.recent(2000)
    if not records:
        return JSONResponse({"error": "No records yet"}, status_code=404)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=drone_session.csv"},
    )


@app.get("/api/model", tags=["config"])
async def get_current_model() -> JSONResponse:
    """Returns the currently active Ollama model name."""
    return JSONResponse({"model": config.OLLAMA_MODEL})


@app.post("/api/config/vision/{enabled}", tags=["config"])
async def toggle_vision(enabled: str) -> JSONResponse:
    """
    Turn the agent's camera (vision) input on/off at runtime.

    On a memory-tight GPU, vision inference can stall; switching to text-only
    keeps the agent fast and reliable (telemetry-only decisions still respect the
    full safety policy).  This is the demo's safety valve.
    """
    on = enabled.lower() in ("1", "true", "on", "yes")
    config.AGENT_USE_VISION = on
    drone_state.add_log("CFG", f"Agent görüş (vision): {'AÇIK' if on else 'KAPALI (sadece telemetri)'}")
    await dashboard_manager.broadcast({"type": "vision_toggle", "enabled": on})
    return JSONResponse({"ok": True, "vision_enabled": on})


@app.get("/api/config/vision", tags=["config"])
async def get_vision_state() -> JSONResponse:
    return JSONResponse({"vision_enabled": config.AGENT_USE_VISION})


@app.post("/api/model/switch/{model_name:path}", tags=["config"])
async def switch_model_endpoint(model_name: str) -> JSONResponse:
    """
    Hot-swap the Ollama model without restarting the server.
    Resets agent singletons — next inference rebuilds with the new model.
    """
    from backend.agent_brain.agent_core import check_ollama, switch_model
    from backend.agent_brain.nav_agent import reset_nav_agent

    switch_model(model_name)
    reset_nav_agent()

    probe = await check_ollama()
    drone_state.add_log("SYS", f"Model değiştirildi: {model_name}")
    await dashboard_manager.broadcast({
        "type": "model_change",
        "model": model_name,
        "model_available": probe.get("model_available", False),
    })
    return JSONResponse({
        "ok": True,
        "model": model_name,
        "reachable": probe.get("reachable", False),
        "model_available": probe.get("model_available", False),
    })


@app.get("/api/models/available", tags=["config"])
async def list_available_models() -> JSONResponse:
    """Lists all vision-capable models currently pulled in Ollama."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            tags = r.json().get("models", [])
        vision_keywords = ["llava", "qwen", "vision", "vl", "minicpm", "gemma3", "moondream"]
        vision = [
            m["name"] for m in tags
            if any(k in m["name"].lower() for k in vision_keywords)
        ]
        return JSONResponse({"models": vision, "active": config.OLLAMA_MODEL})
    except Exception as exc:
        return JSONResponse({"models": [], "active": config.OLLAMA_MODEL, "error": str(exc)})


@app.post("/api/benchmark/quick", tags=["benchmark"])
async def quick_benchmark() -> JSONResponse:
    """
    Quick automated benchmark: runs gradual_tilt scenario for 15s in traditional
    mode, then 15s in agentic mode, then clears and returns comparison stats.
    Non-blocking — returns immediately with a job ID; poll /api/metrics/summary.
    """
    global _demo_task

    async def _run_benchmark():
        orig_mode = drone_state.agent_mode
        drone_state.drone_connected = True
        metrics.clear()

        # Phase 1 — traditional
        drone_state.agent_mode = "traditional"
        await dashboard_manager.broadcast({"type": "mode_change", "mode": "traditional"})
        drone_state.add_log("BNC", "Benchmark başladı — GELENEKSEL modu (15s)")
        t1 = asyncio.create_task(_scenario_gradual_tilt())
        await asyncio.sleep(15)
        t1.cancel()

        # Phase 2 — agentic
        drone_state.agent_mode = "agentic"
        await dashboard_manager.broadcast({"type": "mode_change", "mode": "agentic"})
        drone_state.add_log("BNC", "Benchmark — AGENTİK modu (15s)")
        t2 = asyncio.create_task(_scenario_gradual_tilt())
        await asyncio.sleep(15)
        t2.cancel()

        drone_state.agent_mode = orig_mode
        drone_state.drone_connected = False
        drone_state.add_log("BNC", "Benchmark tamamlandı ✓ — /api/metrics/summary'den sonuçlara bakın")
        await dashboard_manager.broadcast({
            "type": "benchmark_done",
            "summary": metrics.summary(),
        })

    if _demo_task and not _demo_task.done():
        _demo_task.cancel()
    _demo_task = asyncio.create_task(_run_benchmark())
    return JSONResponse({"ok": True, "message": "Benchmark başladı (30s). Sonuç: /api/metrics/summary"})


@app.post("/api/nav/analyze", tags=["navigation"])
async def nav_analyze_endpoint() -> JSONResponse:
    """
    One-shot visual navigation analysis of the current camera frame.
    Returns what the AI sees + recommended action vs what the rule would say.
    This is the 'AI sees, rule is blind' demo endpoint.
    """
    from backend.agent_brain.nav_agent import nav_analyze
    result = await nav_analyze()
    await dashboard_manager.broadcast({
        "type": "nav_result",
        "nav": result,
    })
    return JSONResponse(result)


@app.post("/api/demo/stop", tags=["demo"])
async def stop_demo_scenario() -> JSONResponse:
    """Stop the running demo scenario."""
    global _demo_task
    if _demo_task and not _demo_task.done():
        _demo_task.cancel()
        _demo_task = None
    drone_state.drone_connected = False
    drone_state.latest_telemetry = None   # clear stale values
    drone_state.shadow_decisions["rule"]  = None
    drone_state.shadow_decisions["agent"] = None
    drone_state.add_log("DEMO", "Senaryo durduruldu.")
    await dashboard_manager.broadcast({"type": "drone_status", "connected": False})
    return JSONResponse({"ok": True})


# ── HTML pages ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def serve_dashboard() -> HTMLResponse:
    p = config.FRONTEND_DIR / "dashboard" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(_placeholder("Dashboard Kontrol Paneli", "Phase 3"))


@app.get("/drone", response_class=HTMLResponse, tags=["ui"])
async def serve_drone_client() -> HTMLResponse:
    p = config.FRONTEND_DIR / "drone_client" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(_placeholder("Drone İstemcisi", "Phase 2"))


@app.get("/sunum", response_class=HTMLResponse, tags=["ui"])
async def serve_presentation() -> HTMLResponse:
    """Turkish jury presentation (Phase 6)."""
    p = config.PRESENTATION_DIR / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(_placeholder("Jüri Sunumu", "Phase 6"))


@app.get("/ozet", response_class=HTMLResponse, tags=["ui"])
async def serve_handout() -> HTMLResponse:
    """One-page printable jury handout (A4, Turkish)."""
    p = config.PRESENTATION_DIR / "handout.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(_placeholder("Jüri Özeti", "handout"))


def _placeholder(title: str, phase: str) -> str:
    return (
        f'<!DOCTYPE html><html><body style="background:#0d1117;color:#58a6ff;'
        f'font-family:monospace;display:flex;flex-direction:column;'
        f'align-items:center;justify-content:center;height:100vh;margin:0">'
        f"<h1>&#x1F681; {title}</h1>"
        f'<p style="color:#8b949e">Will be generated in {phase}</p>'
        f'<p><a href="/health" style="color:#3fb950">/health</a></p></body></html>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: Drone Client  (Phone → Server)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/drone")
async def ws_drone(ws: WebSocket) -> None:
    await ws.accept()
    drone_state.drone_connected = True
    drone_state.add_log("DRONE", "Drone client connected ✓")
    await dashboard_manager.broadcast({"type": "drone_status", "connected": True})

    pending_telemetry: dict = {}

    try:
        while True:
            msg = await ws.receive()

            raw: Optional[bytes] = msg.get("bytes")
            if raw:
                frame_b64 = base64.b64encode(raw).decode("ascii")
                drone_state.push_frame(frame_b64, pending_telemetry)
                continue

            text: Optional[str] = msg.get("text")
            if not text:
                continue

            data: dict = json.loads(text)
            t = data.get("type", "")

            if t == "telemetry":
                pending_telemetry = data
                drone_state.note_telemetry(data)

            elif t == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

    except WebSocketDisconnect:
        drone_state.add_log("DRONE", "Drone client disconnected", level="WARNING")
    except json.JSONDecodeError as exc:
        drone_state.add_log("DRONE", f"Malformed JSON: {exc}", level="ERROR")
    except Exception as exc:
        drone_state.add_log("DRONE", f"WS error: {exc}", level="ERROR")
    finally:
        drone_state.drone_connected = False
        await dashboard_manager.broadcast({"type": "drone_status", "connected": False})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: Dashboard  (Server → Browser)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket) -> None:
    await dashboard_manager.connect(ws)
    drone_state.add_log("DASH", "Dashboard connected")

    await ws.send_text(json.dumps({
        "type": "init",
        "mode": drone_state.agent_mode,
        "drone_connected": drone_state.drone_connected,
        "motors": drone_state.motors.to_dict(),
        "logs": drone_state.decision_log[-50:],
    }))

    try:
        while True:
            text = await ws.receive_text()
            cmd: dict = json.loads(text)
            t = cmd.get("type", "")

            if t == "set_mode":
                mode = cmd.get("mode", "traditional")
                if mode in VALID_MODES:
                    drone_state.agent_mode = mode
                    drone_state.add_log("DASH", f"Mode → {mode.upper()}")
                    await dashboard_manager.broadcast({"type": "mode_change", "mode": mode})

            elif t == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("Dashboard WS: %s", exc)
    finally:
        dashboard_manager.disconnect(ws)
        drone_state.add_log("DASH", "Dashboard disconnected", level="WARNING")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Auto-enable HTTPS when the self-signed cert pair is present.  The phone
    # camera (getUserMedia) refuses to run over plain HTTP from a LAN IP, so
    # serving HTTPS here is what makes the live phone demo work out of the box.
    ssl_kwargs: dict = {}
    if config.SSL_CERTFILE.exists() and config.SSL_KEYFILE.exists():
        ssl_kwargs = {
            "ssl_certfile": str(config.SSL_CERTFILE),
            "ssl_keyfile": str(config.SSL_KEYFILE),
        }
        scheme = "https"
        logger.info("TLS certs found → starting with HTTPS (phone camera ready).")
    else:
        scheme = "http"
        logger.warning(
            "No TLS certs (cert.pem/key.pem) → starting with HTTP. "
            "Phone camera will NOT work over the LAN; see PROGRESS.md to generate certs."
        )
    logger.info("Dashboard: %s://<your-ip>:%d/   Drone: %s://<your-ip>:%d/drone",
                scheme, config.PORT, scheme, config.PORT)
    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level="info", **ssl_kwargs)


if __name__ == "__main__":
    main()
