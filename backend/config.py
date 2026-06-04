"""
Central configuration — read from environment variables with sane local defaults.
Copy .env.example to .env and adjust values for your network.
"""
import os
from pathlib import Path

# ── Network ───────────────────────────────────────────────────────────────────
HOST: str = os.getenv("DRONE_HOST", "0.0.0.0")
PORT: int = int(os.getenv("DRONE_PORT", "8000"))

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Default to llava:7b — a vision model that runs on current Ollama builds and on
# 8 GB VRAM. The project spec names "Llama 3.2-Vision or Qwen2.5"; llama3.2-vision
# uses the legacy 'mllama' architecture that newer Ollama builds no longer load
# ("unknown model architecture: 'mllama'"), so llava is the working substitute.
# Override anytime, e.g.  OLLAMA_MODEL=qwen2.5vl:7b  or  OLLAMA_MODEL=llama3.2-vision
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava:7b")

# pydantic-ai talks to Ollama through its OpenAI-compatible endpoint, which lives
# under /v1.  Derive it once here so the rest of the code never has to remember.
OLLAMA_OPENAI_BASE_URL: str = os.getenv(
    "OLLAMA_OPENAI_BASE_URL",
    OLLAMA_BASE_URL.rstrip("/") + "/v1",
)

# Per-request timeout for the LLM call (seconds).  Vision models on CPU can be
# slow, so this is generous; the agent loop will skip a tick rather than block
# forever.
OLLAMA_TIMEOUT_S: float = float(os.getenv("OLLAMA_TIMEOUT_S", "60.0"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
PRESENTATION_DIR: Path = PROJECT_ROOT / "presentation"
LOG_DIR: Path = PROJECT_ROOT / "logs"
BENCHMARK_DIR: Path = PROJECT_ROOT / "benchmarks" / "results"

# ── TLS / HTTPS (phone camera needs a secure context) ───────────────────────────
# navigator.mediaDevices.getUserMedia() only works over HTTPS when the page is
# served to a phone over the LAN (non-localhost origin).  If these cert files
# exist, ``drone-server`` starts with HTTPS automatically — no extra flags needed.
SSL_CERTFILE: Path = Path(os.getenv("DRONE_SSL_CERT", str(PROJECT_ROOT / "cert.pem")))
SSL_KEYFILE: Path = Path(os.getenv("DRONE_SSL_KEY", str(PROJECT_ROOT / "key.pem")))

# ── Agent / Broadcast timing ──────────────────────────────────────────────────
BROADCAST_HZ: float = 10.0                          # Dashboard push rate
BROADCAST_INTERVAL: float = 1.0 / BROADCAST_HZ

AGENT_INFERENCE_HZ: float = 1.0                     # LLM call rate (expensive)
AGENT_INFERENCE_INTERVAL: float = 1.0 / AGENT_INFERENCE_HZ

MAX_DECISION_LOG: int = 200                          # Rolling log cap

# ── Rule-based thresholds (Phase 3 / Traditional mode) ───────────────────────
TILT_DANGER_DEG: float = float(os.getenv("TILT_DANGER_DEG", "30.0"))
TILT_WARNING_DEG: float = float(os.getenv("TILT_WARNING_DEG", "15.0"))
MOTOR_DEFAULT: int = int(os.getenv("MOTOR_DEFAULT", "50"))    # 0-100 %
MOTOR_STABILIZE: int = int(os.getenv("MOTOR_STABILIZE", "75"))
MOTOR_EMERGENCY: int = int(os.getenv("MOTOR_EMERGENCY", "100"))

# ── Agentic mode (Phase 4) ───────────────────────────────────────────────────
# Number of times the agent retries structured-output validation before giving
# up on a tick (pydantic-ai re-prompts the model on a validation failure).
AGENT_OUTPUT_RETRIES: int = int(os.getenv("AGENT_OUTPUT_RETRIES", "2"))

# When True the agent attaches the latest camera frame to every prompt.  Set to
# False to run the agent on telemetry alone (useful with non-vision models or to
# isolate the cost of vision during benchmarking).
AGENT_USE_VISION: bool = os.getenv("AGENT_USE_VISION", "true").lower() == "true"

# ── Short-term memory (multi-frame context) ──────────────────────────────────
# How many recent telemetry samples the agent keeps as a rolling "short memory".
# A trend summary (rising / falling tilt) computed from these is added to the
# prompt so the agent can reason about motion over time, not just a single frame.
# Set to 1 to disable the trend (single-frame behaviour).
AGENT_MEMORY_FRAMES: int = int(os.getenv("AGENT_MEMORY_FRAMES", "5"))

# ── Hybrid mode (reflex + deliberation) ──────────────────────────────────────
# The hybrid controller runs the deterministic rule engine as a fast REFLEX
# layer that guarantees safety every tick, while the LLM agent runs as a slower
# DELIBERATION layer that refines behaviour only when the situation is safe.
# Reflex cadence: fast, sub-ms, keeps the drone safe continuously.
HYBRID_REFLEX_HZ: float = float(os.getenv("HYBRID_REFLEX_HZ", "10.0"))
HYBRID_REFLEX_INTERVAL: float = 1.0 / HYBRID_REFLEX_HZ
# A cached deliberation older than this many seconds is treated as stale and the
# reflex layer takes over until the agent produces a fresh decision.
HYBRID_DELIBERATION_TTL_S: float = float(os.getenv("HYBRID_DELIBERATION_TTL_S", "5.0"))

# Down-scale frames before sending to the vision model — smaller images are much
# faster to encode/transfer and keep the prompt within the model's context.
AGENT_VISION_MAX_EDGE: int = int(os.getenv("AGENT_VISION_MAX_EDGE", "512"))

# Rolling buffer of per-decision metric records kept in memory for /api/metrics.
METRICS_BUFFER_SIZE: int = int(os.getenv("METRICS_BUFFER_SIZE", "500"))

# ── Structured logging (Phase 5) ─────────────────────────────────────────────
# Every decision (rule or agent) is appended as one JSON object per line.
DECISION_LOG_FILE: str = os.getenv(
    "DECISION_LOG_FILE", str(LOG_DIR / "decisions.jsonl")
)
ENABLE_FILE_LOGGING: bool = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
