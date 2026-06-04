"""
Drone Brain — FastMCP Server (Phase 3)
=======================================
Mounted by main.py at  /mcp  (SSE transport).
Phase-5 agent connects via:  http://localhost:8000/mcp/sse

Resources  →  read-only sensor snapshots for the LLM
Tools      →  actuator commands the LLM can call autonomously
Rule engine→  synchronous IF/THEN fallback (called by main.py background task)
"""
import logging
from typing import Any

from fastmcp import FastMCP

from backend import config
from backend.state import drone_state

logger = logging.getLogger("drone.mcp")

# ─────────────────────────────────────────────────────────────────────────────
# FastMCP instance
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="Drone Brain MCP",
    instructions=(
        "You are the autonomous flight controller for a smartphone drone simulation. "
        "Read sensor data via resources, then issue motor commands via tools. "
        "Always explain your reasoning before calling a tool. "
        "If a sensor is unavailable, call report_sensor_failure instead of guessing."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Resources  (read-only — exposed to the LLM as context)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("drone://camera/latest-frame")
async def camera_frame_resource() -> str:
    """
    Latest camera frame from the drone client, base64-encoded JPEG.
    Returns a data-URL string that Ollama's vision endpoint can parse directly.
    Empty string when no frame is available yet.
    """
    frame = drone_state.latest_frame_b64
    if not frame:
        return ""
    return f"data:image/jpeg;base64,{frame}"


@mcp.resource("drone://telemetry/imu")
async def telemetry_imu_resource() -> dict:
    """
    Latest IMU reading from the drone client.

    Keys:
      ax, ay, az  — linear acceleration  (m/s²)
      alpha       — yaw   (°, 0–360)
      beta        — pitch (°, −180 to +180)
      gamma       — roll  (°, −90 to +90)
      ts          — Unix timestamp of the reading
    """
    tele = drone_state.latest_telemetry
    if not tele:
        return {"error": "no telemetry received yet"}
    return tele


@mcp.resource("drone://motors/state")
async def motor_state_resource() -> dict:
    """
    Current virtual motor speeds and flight mode.

    Keys:
      fl, fr, rl, rr  — motor speeds 0–100 %
                         (front-left, front-right, rear-left, rear-right)
      mode            — current flight mode string
    """
    return drone_state.motors.to_dict()


@mcp.resource("drone://system/status")
async def system_status_resource() -> dict:
    """System overview: connection state, active mode, log count."""
    return {
        "drone_connected": drone_state.drone_connected,
        "agent_mode": drone_state.agent_mode,
        "log_count": len(drone_state.decision_log),
        "last_log": drone_state.decision_log[-1] if drone_state.decision_log else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tools  (actuators — the LLM calls these to issue commands)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def adjust_motor_speed(motor_id: str, speed: int) -> dict:
    """
    Set the speed of one or all virtual motors.

    Args:
        motor_id: "fl" | "fr" | "rl" | "rr" | "all"
        speed:    Target speed 0–100 (%)
    """
    speed = max(0, min(100, int(speed)))
    motors = drone_state.motors
    valid = {"fl", "fr", "rl", "rr", "all"}

    if motor_id not in valid:
        msg = f"Unknown motor_id '{motor_id}'. Valid: {sorted(valid)}"
        drone_state.add_log("MCP", msg, level="ERROR")
        return {"ok": False, "error": msg}

    if motor_id == "all":
        motors.set_all(speed, mode="manual")
    else:
        setattr(motors, motor_id, speed)
        motors.mode = "manual"

    drone_state.add_log("MCP", f"Motor {motor_id.upper()} → {speed}%")
    return {"ok": True, "motor_id": motor_id, "speed": speed, "motors": motors.to_dict()}


@mcp.tool()
async def stabilize_drone() -> dict:
    """
    Stabilization response: equalise all motors to MOTOR_STABILIZE speed.
    Call when tilt is between warning and danger thresholds.
    """
    motors = drone_state.motors
    motors.set_all(config.MOTOR_STABILIZE, mode="stabilize")
    drone_state.add_log(
        "MCP",
        f"STABILIZE — all motors → {config.MOTOR_STABILIZE}%",
        level="WARNING",
    )
    return {"ok": True, "action": "stabilize", "motors": motors.to_dict()}


@mcp.tool()
async def emergency_stop() -> dict:
    """
    Cut all motor power immediately.
    Use only when extreme danger is detected and no recovery is possible.
    """
    drone_state.motors.set_all(0, mode="emergency_stop")
    drone_state.add_log("MCP", "EMERGENCY STOP — all motors → 0%", level="ERROR")
    return {"ok": True, "action": "emergency_stop", "motors": drone_state.motors.to_dict()}


@mcp.tool()
async def turn_left(degrees: float = 15.0) -> dict:
    """
    Yaw left: reduce left-side motors, increase right-side.

    Args:
        degrees: Target yaw angle (° — used for proportional mapping).
    """
    m = drone_state.motors
    base  = config.MOTOR_DEFAULT
    delta = max(1, int(degrees * 0.5))
    m.fl  = max(0,   base - delta)
    m.rl  = max(0,   base - delta)
    m.fr  = min(100, base + delta)
    m.rr  = min(100, base + delta)
    m.mode = "turn_left"
    drone_state.add_log("MCP", f"TURN LEFT {degrees:.1f}°")
    return {"ok": True, "action": "turn_left", "degrees": degrees, "motors": m.to_dict()}


@mcp.tool()
async def turn_right(degrees: float = 15.0) -> dict:
    """
    Yaw right: increase left-side motors, reduce right-side.

    Args:
        degrees: Target yaw angle (°).
    """
    m = drone_state.motors
    base  = config.MOTOR_DEFAULT
    delta = max(1, int(degrees * 0.5))
    m.fl  = min(100, base + delta)
    m.rl  = min(100, base + delta)
    m.fr  = max(0,   base - delta)
    m.rr  = max(0,   base - delta)
    m.mode = "turn_right"
    drone_state.add_log("MCP", f"TURN RIGHT {degrees:.1f}°")
    return {"ok": True, "action": "turn_right", "degrees": degrees, "motors": m.to_dict()}


@mcp.tool()
async def set_hover() -> dict:
    """Return to balanced hover: set all motors to MOTOR_DEFAULT speed."""
    drone_state.motors.set_all(config.MOTOR_DEFAULT, mode="hover")
    drone_state.add_log("MCP", f"HOVER — all motors → {config.MOTOR_DEFAULT}%")
    return {"ok": True, "action": "hover", "motors": drone_state.motors.to_dict()}


@mcp.tool()
async def report_sensor_failure(sensor: str, reason: str) -> dict:
    """
    Log a sensor failure without crashing the agent loop.
    Call this instead of raising an exception when a sensor is unavailable.

    Args:
        sensor: Sensor name (e.g. "accelerometer", "camera", "gyroscope").
        reason: Human-readable description of the failure.
    """
    msg = f"SENSOR FAILURE [{sensor}]: {reason}"
    drone_state.add_log("MCP", msg, level="ERROR")
    return {"ok": True, "logged": True, "sensor": sensor, "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Traditional IF/THEN Rule Engine
# Called by the background task in main.py when mode == "traditional".
# NOT exposed as an MCP tool — it bypasses the LLM entirely.
# ─────────────────────────────────────────────────────────────────────────────

def compute_rule_decision_only() -> dict[str, Any]:
    """
    Compute the IF/THEN rule decision without any side effects.
    No motor updates, no log entries — pure read + compute.
    Used by the shadow tracker so the compare strip always shows a rule verdict
    even when the active mode is 'agentic' or 'hybrid'.
    """
    tele = drone_state.latest_telemetry
    if not tele:
        return {
            "rule": "NO_DATA", "action": "HOLD", "tilt_deg": 0.0,
            "hazard": "NONE", "confidence": 1.0,
            "motor_speed": config.MOTOR_DEFAULT,
            "reasoning": "Telemetri verisi henüz yok",
        }
    beta  = abs(float(tele.get("beta",  0.0) or 0.0))
    gamma = abs(float(tele.get("gamma", 0.0) or 0.0))
    tilt  = max(beta, gamma)
    if tilt >= config.TILT_DANGER_DEG:
        return {
            "rule": "EMERGENCY_STABILIZE", "action": "STABILIZE", "tilt_deg": tilt,
            "hazard": "CRITICAL", "confidence": 1.0,
            "motor_speed": config.MOTOR_EMERGENCY,
            "reasoning": (
                f"IF tilt({tilt:.1f}°) ≥ {config.TILT_DANGER_DEG:.0f}° "
                f"→ EMERGENCY_STABILIZE @ {config.MOTOR_EMERGENCY}%"
            ),
        }
    if tilt >= config.TILT_WARNING_DEG:
        return {
            "rule": "STABILIZE", "action": "STABILIZE", "tilt_deg": tilt,
            "hazard": "MEDIUM", "confidence": 1.0,
            "motor_speed": config.MOTOR_STABILIZE,
            "reasoning": (
                f"IF tilt({tilt:.1f}°) ≥ {config.TILT_WARNING_DEG:.0f}° "
                f"→ STABILIZE @ {config.MOTOR_STABILIZE}%"
            ),
        }
    return {
        "rule": "HOVER", "action": "HOVER", "tilt_deg": tilt,
        "hazard": "NONE", "confidence": 1.0,
        "motor_speed": config.MOTOR_DEFAULT,
        "reasoning": (
            f"tilt({tilt:.1f}°) < {config.TILT_WARNING_DEG:.0f}° → HOVER @ {config.MOTOR_DEFAULT}%"
        ),
    }


async def run_traditional_rules() -> dict[str, Any]:
    """
    Synchronous IF/THEN rule engine — no LLM involvement.

    Decision hierarchy (highest priority first):
      1. No telemetry yet              → HOLD (log warning)
      2. |tilt| ≥ TILT_DANGER_DEG     → EMERGENCY_STABILIZE
      3. |tilt| ≥ TILT_WARNING_DEG    → STABILIZE
      4. Nominal                       → HOVER

    'tilt' is defined as max(|beta|, |gamma|) — worst-axis tilt angle.

    Returns a result dict for the caller's log/telemetry.
    """
    tele   = drone_state.latest_telemetry
    motors = drone_state.motors

    # ── Rule 0: No data ───────────────────────────────────────────────────────
    if not tele:
        drone_state.add_log(
            "RULE",
            "No telemetry — holding last motor state",
            level="WARNING",
            extra={"action": "HOLD", "hazard": "NONE", "confidence": 1.0,
                   "used_vision": False, "ok": False},
        )
        return {"rule": "NO_DATA", "action": "hold", "motors": motors.to_dict()}

    beta  = abs(float(tele.get("beta",  0.0)))
    gamma = abs(float(tele.get("gamma", 0.0)))
    tilt  = max(beta, gamma)

    # ── Rule 1: Emergency tilt ────────────────────────────────────────────────
    if tilt >= config.TILT_DANGER_DEG:
        motors.set_all(config.MOTOR_EMERGENCY, mode="emergency_stabilize")
        drone_state.add_log(
            "RULE",
            (
                f"IF tilt({tilt:.1f}°) ≥ {config.TILT_DANGER_DEG}° "
                f"→ EMERGENCY_STABILIZE @ {config.MOTOR_EMERGENCY}%"
            ),
            level="ERROR",
            extra={"action": "STABILIZE", "hazard": "HIGH", "confidence": 1.0,
                   "used_vision": False, "ok": True, "tilt_deg": round(tilt, 1)},
        )
        return {
            "rule": "EMERGENCY_STABILIZE",
            "tilt_deg": tilt,
            "motor_speed": config.MOTOR_EMERGENCY,
            "motors": motors.to_dict(),
        }

    # ── Rule 2: Warning tilt ──────────────────────────────────────────────────
    if tilt >= config.TILT_WARNING_DEG:
        motors.set_all(config.MOTOR_STABILIZE, mode="stabilize")
        drone_state.add_log(
            "RULE",
            (
                f"IF tilt({tilt:.1f}°) ≥ {config.TILT_WARNING_DEG}° "
                f"→ STABILIZE @ {config.MOTOR_STABILIZE}%"
            ),
            level="WARNING",
            extra={"action": "STABILIZE", "hazard": "MEDIUM", "confidence": 1.0,
                   "used_vision": False, "ok": True, "tilt_deg": round(tilt, 1)},
        )
        return {
            "rule": "STABILIZE",
            "tilt_deg": tilt,
            "motor_speed": config.MOTOR_STABILIZE,
            "motors": motors.to_dict(),
        }

    # ── Rule 3: Nominal ───────────────────────────────────────────────────────
    motors.set_all(config.MOTOR_DEFAULT, mode="hover")
    drone_state.add_log(
        "RULE",
        (
            f"IF tilt({tilt:.1f}°) < {config.TILT_WARNING_DEG}° "
            f"→ HOVER @ {config.MOTOR_DEFAULT}%"
        ),
        extra={"action": "HOVER", "hazard": "NONE", "confidence": 1.0,
               "used_vision": False, "ok": True, "tilt_deg": round(tilt, 1)},
    )
    return {
        "rule": "HOVER",
        "tilt_deg": tilt,
        "motor_speed": config.MOTOR_DEFAULT,
        "motors": motors.to_dict(),
    }
