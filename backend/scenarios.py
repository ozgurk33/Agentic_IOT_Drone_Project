"""
Synthetic flight scenarios for benchmarking & testing
=====================================================
A *scenario* is a deterministic sequence of IMU telemetry samples (and, on
demand, matching camera frames) that exercises the controllers without a real
phone in the air. The same scenarios drive both the Phase-5 benchmark and the
pytest suite, so behaviour is reproducible and comparable.

Each step is a telemetry dict shaped exactly like what the phone streams:
    {alpha, beta, gamma, ax, ay, az, ts}
where beta=pitch, gamma=roll (degrees). ``tilt = max(|beta|, |gamma|)``.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Scenario:
    name: str
    description: str
    steps: list[dict]

    def __len__(self) -> int:
        return len(self.steps)


def _tele(beta: float, gamma: float, *, alpha: float = 0.0,
          ax: float = 0.0, ay: float = 0.0, az: float = 9.81, ts: float = 0.0) -> dict:
    return {"alpha": alpha, "beta": beta, "gamma": gamma,
            "ax": ax, "ay": ay, "az": az, "ts": ts, "type": "telemetry"}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ─────────────────────────────────────────────────────────────────────────────

def stable_hover(n: int = 20) -> Scenario:
    """Calm flight: tiny jitter, always well below the warning threshold."""
    steps = []
    for i in range(n):
        wob = 2.0 * math.sin(i / 3.0)
        steps.append(_tele(beta=wob, gamma=wob * 0.6, alpha=90.0, ts=float(i)))
    return Scenario("stable_hover", "Sakin uçuş — eğim sürekli güvenli bölgede.", steps)


def gradual_tilt(n: int = 24) -> Scenario:
    """Tilt ramps linearly from 0° up to ~40° — crosses warning then danger."""
    steps = []
    for i in range(n):
        t = (i / (n - 1)) * 40.0
        steps.append(_tele(beta=t, gamma=t * 0.3, alpha=45.0, ax=t * 0.05, ts=float(i)))
    return Scenario("gradual_tilt", "Eğim 0°→40° kademeli artıyor — uyarı ve tehlike eşikleri aşılır.", steps)


def sudden_gust(n: int = 20) -> Scenario:
    """Calm, a violent gust spikes tilt past danger, then a recovery."""
    steps = []
    for i in range(n):
        if i < 6:
            beta = 3.0
        elif i < 10:
            beta = 38.0          # gust
        else:
            beta = max(3.0, 38.0 - (i - 10) * 5.0)   # recovery
        steps.append(_tele(beta=beta, gamma=4.0, alpha=120.0,
                           ax=(beta / 10.0), ts=float(i)))
    return Scenario("sudden_gust", "Ani rüzgar darbesi eğimi tehlikeye fırlatır, ardından toparlanma.", steps)


def oscillation(n: int = 30) -> Scenario:
    """Sinusoidal tilt oscillating around the warning threshold."""
    steps = []
    for i in range(n):
        beta = 16.0 + 14.0 * math.sin(i / 2.0)   # swings ~2°..30°
        steps.append(_tele(beta=beta, gamma=6.0 * math.cos(i / 2.0),
                           alpha=200.0, ts=float(i)))
    return Scenario("oscillation", "Uyarı eşiği etrafında salınan eğim — sınır davranışı testi.", steps)


def sensor_dropout(n: int = 18) -> Scenario:
    """Some steps carry no telemetry at all (sensor dropout / bad packet)."""
    steps = []
    for i in range(n):
        if i in (4, 5, 11):
            steps.append({})                 # dropout
        else:
            steps.append(_tele(beta=10.0 + (i % 5) * 4.0, gamma=5.0, ts=float(i)))
    return Scenario("sensor_dropout", "Aralıklı sensör kesintisi — eksik veriye dayanıklılık testi.", steps)


ALL_BUILDERS: dict[str, Callable[[], Scenario]] = {
    "stable_hover": stable_hover,
    "gradual_tilt": gradual_tilt,
    "sudden_gust": sudden_gust,
    "oscillation": oscillation,
    "sensor_dropout": sensor_dropout,
}


def all_scenarios() -> list[Scenario]:
    return [build() for build in ALL_BUILDERS.values()]


def get_scenario(name: str) -> Scenario:
    if name not in ALL_BUILDERS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(ALL_BUILDERS)}")
    return ALL_BUILDERS[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic camera frame (a tilted artificial horizon) — for vision benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def render_frame_b64(telemetry: dict, size: int = 256) -> Optional[str]:
    """
    Render a simple artificial-horizon JPEG whose horizon line is rotated by the
    drone's roll, returned as base64 (no data-URL prefix) — exactly the format
    the live phone client streams.

    Returns None if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    roll = float(telemetry.get("gamma", 0.0) or 0.0)
    pitch = float(telemetry.get("beta", 0.0) or 0.0)

    img = Image.new("RGB", (size, size), (135, 206, 235))   # sky
    draw = ImageDraw.Draw(img)

    # Horizon offset by pitch, rotated by roll.
    cx, cy = size / 2, size / 2 + pitch * 2.0
    angle = math.radians(roll)
    dx, dy = math.cos(angle), math.sin(angle)
    length = size * 1.5
    p1 = (cx - dx * length, cy - dy * length)
    p2 = (cx + dx * length, cy + dy * length)
    # Ground polygon below the horizon line.
    draw.polygon(
        [p1, p2, (size * 2, size * 2), (-size, size * 2)],
        fill=(110, 90, 60),
    )
    draw.line([p1, p2], fill=(255, 255, 255), width=3)
    # Centre reticle (the drone's fixed reference).
    draw.line([(cx - 30, size / 2), (cx + 30, size / 2)], fill=(255, 0, 0), width=3)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")
