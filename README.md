# Drone MCP Project

**Smartphone as Autonomous Drone Simulation — Rule-Based vs Agentic AI**

An academic project that turns a phone into a "drone body" (camera + IMU streamed
over WebSocket) and a Kubuntu PC into its "brain". The same flight-control task is
solved three ways and compared head-to-head:

- **Traditional** — a deterministic IF/THEN rule engine.
- **Agentic** — a local Pydantic-AI agent driving an Ollama vision model
  (`llava:7b`), deciding via structured output and acting through MCP tools.
- **Hybrid** — a subsumption controller: the rule engine is a fast reflex layer
  that guarantees safety every tick (<1 ms), while the agent deliberates
  asynchronously and refines behaviour only when the situation is safe. Gets
  rule-speed in danger and agent-context when calm; never crashes if Ollama is down.

Everything runs **locally** — no cloud LLM APIs.

---

## Architecture

```
Phone (browser)              Kubuntu PC
─────────────────            ─────────────────────────────────────────────
camera + IMU  ──WS──►  FastAPI WebSocket Hub ──►  DroneState (frame_queue=1)
                              │  ▲                        │
                       /ws/dashboard (10 Hz)             ├─► Rule Engine (IF/THEN)
                              ▼                            │
                        Dashboard UI                       └─► Pydantic-AI Agent
                                                                  │   ▲
                                                       DroneDecision   │
                                                                  ▼   │
                                                          MCP Tools  Ollama (/v1)
```

| Phase | What | Status |
|------|------|--------|
| 1–2 | Phone client + WebSocket streaming | ✅ |
| 3 | MCP server + dashboard + rule engine | ✅ |
| 4 | Pydantic-AI + Ollama autonomous agent loop | ✅ |
| 5 | Benchmark, test suite, structured logging | ✅ |
| 6 | Turkish HTML5 jury presentation | ✅ |
| + | Hybrid controller (reflex + deliberation) + multi-frame memory | ✅ |

---

## Quick start

```bash
# 1. Install deps
uv sync

# 2. (Agentic mode) make sure Ollama is running and the model is pulled
ollama serve &
ollama pull llava:7b          # verified default (vision works on current Ollama)
# ollama pull qwen2.5vl:7b    # recommended for obstacle detection / navigation

# 3. Run the server (HTTPS auto-enables when cert.pem/key.pem exist —
#    REQUIRED for the phone camera over the LAN)
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem

# 4. (Optional) verify the whole pipeline before a demo
python scripts/verify_e2e.py https://localhost:8000   # expect 8/8 PASS
```

Then open:

| URL | Page |
|-----|------|
| `http://<pc-ip>:8000/`        | Dashboard / control panel |
| `http://<pc-ip>:8000/drone`   | Mobile drone client (open on the phone) |
| `http://<pc-ip>:8000/sunum`   | **Jury presentation** (dark theme, Turkish) |
| `http://localhost:8000/health`| Health check |
| `http://localhost:8000/api/metrics/summary` | Live per-mode metrics |
| `http://localhost:8000/api/ollama` | Ollama reachability + model probe |

Switch modes from the dashboard, or:
`curl -X POST http://localhost:8000/api/mode/agentic` (or `/traditional`, `/hybrid`)

> If Ollama is unreachable, agentic mode automatically falls back to a safe
> rule-based action — the loop never crashes.

---

## Dashboard demo features

The control panel is built for a live jury demo:

- **Side-by-side compare strip** — rule vs agent decisions at all times (both run
  continuously as "shadow" decisions), showing the exact `IF tilt ≥ X°` condition,
  the agent's reasoning, latency, and a 📷 camera-used indicator.
- **Paradigm divergence banner** — flashes the moment rule and agent decisions differ.
- **Flight HUD** over the camera — artificial horizon, pitch ladder, roll arc, heading.
- **3D drone view** — tilts with the phone, rotor speed = motor %.
- **Live tilt chart** — 60-second history with threshold lines + decision dots.
- **Latency race + session report** — per-mode min/avg/max, action mix, accuracy.
- **Visual navigation** (`🎯 NAVİGASYON`) — "rule is blind, AI sees": the AI detects
  obstacles from the camera while the tilt-only rule says HOVER.
- **Demo scenarios** (no phone needed): calm, gradual tilt, shake, critical,
  turbulence, wind gust, zigzag, obstacle.
- **Live controls** — model switcher (`llava` / `qwen2.5vl` / `llama3.2-vision`),
  vision on/off safety valve, runtime threshold editor, quick 30 s benchmark,
  CSV export, sound alerts, presentation flow, toasts, boot splash.

**Keyboard shortcuts (dashboard):** `1/2/3` modes · `N` nav · `Q/W/E/R` scenarios ·
`X` stop · `A` analyze · `P` presentation · `B` benchmark · `M` report · `T` thresholds ·
`V` vision · `S` sound · `F` fullscreen · `?` help.

**Presentation (`/sunum`):** `←/→` navigate · `T` reset timer · `N` speaker notes ·
`O` slide overview · `F` fullscreen.

---

## Benchmark (Phase 5)

Compares both controllers over reproducible synthetic flight scenarios and writes
`benchmarks/results/<timestamp>_{records.jsonl,records.csv,summary.json}`.

```bash
uv run python scripts/benchmark.py            # auto-detect Ollama
uv run python scripts/benchmark.py --model mock      # offline mock LLM (no GPU)
uv run python scripts/benchmark.py --model ollama --vision --repeat 3

# Analyse any decisions JSONL log afterwards:
uv run python scripts/analyze_logs.py logs/decisions.jsonl
```

The **mock model** lets the whole pipeline (prompt → structured output → MCP
dispatch → metrics) run end-to-end without a GPU, so tests and benchmarks work on
any machine.

---

## Tests

```bash
uv run pytest          # 46 tests: rules, agent, hybrid, memory, dispatch, metrics, scenarios, state
```

---

## Layout

```
backend/
  config.py            env-configurable constants
  main.py              FastAPI app, WS hub, background loops, REST API
  state.py             DroneState / MotorState singleton
  metrics.py           DecisionRecord + MetricsCollector (ring buffer + JSONL)
  scenarios.py         synthetic flight scenarios + artificial-horizon frames
  mcp_server/          FastMCP resources, tools, rule engine (Phase 3)
  agent_brain/
    agent_core.py      Pydantic-AI agent, DroneDecision, dispatch, loop (Phase 4)
    mock_model.py      offline FunctionModel substitute for Ollama
frontend/
  dashboard/           PC control panel (Phase 3)
  drone_client/        mobile HTML5 client (Phase 2)
presentation/
  index.html           Turkish jury slide deck (Phase 6)
scripts/
  benchmark.py         rule-based vs agentic comparison
  analyze_logs.py      JSONL log summariser
tests/                 pytest suite
```

Configuration: copy `.env.example` → `.env` and adjust for your network/model.
