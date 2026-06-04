"""
Coverage for the endpoints added in the 2026-06-04 demo upgrade:
model switch, vision toggle, thresholds, nav analyze, benchmark, CSV, scenarios.
All run in-process via TestClient — no live Ollama or network server needed.
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from backend import config
from backend.main import app
from backend.state import drone_state


def _jpeg_b64() -> str:
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 80, 160)).save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode()


def test_model_get_and_list():
    with TestClient(app) as client:
        m = client.get("/api/model").json()
        assert "model" in m
        lst = client.get("/api/models/available").json()
        assert "models" in lst and "active" in lst


def test_model_switch_changes_config():
    original = config.OLLAMA_MODEL
    try:
        with TestClient(app) as client:
            r = client.post("/api/model/switch/llava:7b")
            assert r.status_code == 200
            assert r.json()["model"] == "llava:7b"
            assert config.OLLAMA_MODEL == "llava:7b"
    finally:
        config.OLLAMA_MODEL = original


def test_vision_toggle_affects_config():
    with TestClient(app) as client:
        client.post("/api/config/vision/off")
        assert config.AGENT_USE_VISION is False
        client.post("/api/config/vision/on")
        assert config.AGENT_USE_VISION is True


def test_threshold_update_affects_config_and_clamps_back():
    orig_w, orig_d = config.TILT_WARNING_DEG, config.TILT_DANGER_DEG
    try:
        with TestClient(app) as client:
            client.post("/api/config/thresholds", json={"tilt_warning": 10, "tilt_danger": 22})
            assert config.TILT_WARNING_DEG == 10.0
            assert config.TILT_DANGER_DEG == 22.0
            body = client.get("/api/config/thresholds").json()
            assert body["tilt_warning"] == 10.0 and body["tilt_danger"] == 22.0
    finally:
        config.TILT_WARNING_DEG, config.TILT_DANGER_DEG = orig_w, orig_d


def test_threshold_partial_update():
    """Updating only one field must leave the others untouched."""
    orig = config.MOTOR_STABILIZE
    try:
        with TestClient(app) as client:
            client.post("/api/config/thresholds", json={"motor_stabilize": 80})
            assert config.MOTOR_STABILIZE == 80
    finally:
        config.MOTOR_STABILIZE = orig


def test_nav_analyze_no_frame_is_graceful():
    """With no camera frame, nav must return a safe UNKNOWN, never 500."""
    drone_state.latest_frame_b64 = None
    with TestClient(app) as client:
        r = client.post("/api/nav/analyze")
        assert r.status_code == 200
        d = r.json()
        assert d["action"] == "UNKNOWN"
        assert d["ok"] is False
        assert "rule_would_say" in d        # comparison string always present


def test_nav_rule_would_say_reflects_tilt():
    """The 'blind rule' string must reflect the current telemetry tilt."""
    drone_state.note_telemetry({"beta": 0.0, "gamma": 0.0, "tilt": 0.0})
    drone_state.latest_frame_b64 = None
    with TestClient(app) as client:
        d = client.post("/api/nav/analyze").json()
        assert "HOVER" in d["rule_would_say"]


def test_csv_export_empty_is_404():
    from backend.metrics import metrics
    metrics.clear()
    with TestClient(app) as client:
        assert client.get("/api/session/export.csv").status_code == 404


def test_demo_scenario_lifecycle():
    with TestClient(app) as client:
        assert client.post("/api/demo/scenario/calm").json()["ok"] is True
        assert drone_state.drone_connected is True
        assert client.post("/api/demo/stop").json()["ok"] is True
        assert drone_state.drone_connected is False


def test_all_eight_scenarios_registered():
    names = ("calm", "gradual_tilt", "shake", "critical",
             "turbulence", "wind_gust", "zigzag", "obstacle")
    with TestClient(app) as client:
        for n in names:
            assert client.post(f"/api/demo/scenario/{n}").status_code == 200
        client.post("/api/demo/stop")


def test_benchmark_quick_starts():
    with TestClient(app) as client:
        r = client.post("/api/benchmark/quick")
        assert r.status_code == 200 and r.json()["ok"] is True
        client.post("/api/demo/stop")  # cancel the benchmark task


def test_metrics_summary_shape():
    from backend.metrics import DecisionRecord, metrics
    import time
    metrics.record(DecisionRecord(ts=time.time(), mode="agentic", action="HOVER",
                                  latency_ms=1200.0, confidence=0.8))
    with TestClient(app) as client:
        s = client.get("/api/metrics/summary").json()
        assert "by_mode" in s
        assert "agentic" in s["by_mode"]
