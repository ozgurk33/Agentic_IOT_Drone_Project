"""
In-process integration tests — exercise the full FastAPI app via TestClient
(no network server, no SSL, no Ollama).  Verifies routing, the WebSocket frame
path, the rule decision pipeline, and the new demo/config endpoints.

These run as part of the normal `pytest` suite.
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.state import drone_state


def _jpeg(color=(40, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def test_health_and_pages():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200          # dashboard
        assert client.get("/drone").status_code == 200      # phone client
        assert client.get("/sunum").status_code == 200      # presentation
        assert client.get("/ozet").status_code == 200        # jury handout


def test_mode_switching():
    with TestClient(app) as client:
        for mode in ("traditional", "agentic", "hybrid"):
            r = client.post(f"/api/mode/{mode}")
            assert r.status_code == 200 and r.json()["mode"] == mode
        assert client.post("/api/mode/bogus").status_code == 400


def test_threshold_editor_roundtrip():
    with TestClient(app) as client:
        client.post("/api/config/thresholds", json={"tilt_danger": 25})
        assert client.get("/api/config/thresholds").json()["tilt_danger"] == 25.0
        # restore
        client.post("/api/config/thresholds", json={"tilt_danger": 30})
        assert client.get("/api/config/thresholds").json()["tilt_danger"] == 30.0


def test_vision_toggle():
    with TestClient(app) as client:
        assert client.post("/api/config/vision/off").json()["vision_enabled"] is False
        assert client.get("/api/config/vision").json()["vision_enabled"] is False
        assert client.post("/api/config/vision/on").json()["vision_enabled"] is True


def test_demo_scenarios_exist():
    with TestClient(app) as client:
        for name in ("calm", "gradual_tilt", "shake", "critical",
                     "turbulence", "wind_gust", "zigzag", "obstacle"):
            r = client.post(f"/api/demo/scenario/{name}")
            assert r.status_code == 200, name
            assert r.json()["ok"] is True
        client.post("/api/demo/stop")
        assert client.post("/api/demo/scenario/nonexistent").status_code == 400


def test_websocket_frame_and_telemetry_path():
    """Phone → server: a binary JPEG + telemetry must populate drone_state."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/drone") as ws:
            ws.send_json({"type": "telemetry", "beta": 35.0, "gamma": 5.0,
                          "tilt": 35.0, "az": 9.8})
            ws.send_bytes(_jpeg())
            # allow the receive loop to process
            ws.send_json({"type": "ping"})
            ws.receive_json()  # pong
        assert drone_state.latest_frame_b64 is not None
        assert drone_state.latest_telemetry.get("tilt") == 35.0


def test_dashboard_receives_init():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/dashboard") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert "mode" in msg and "motors" in msg


def test_csv_export_after_decisions():
    """After at least one decision is recorded, CSV export returns rows."""
    import time

    from backend.metrics import DecisionRecord, metrics

    metrics.record(DecisionRecord(ts=time.time(), mode="traditional",
                                  action="STABILIZE", latency_ms=0.1))
    with TestClient(app) as client:
        r = client.get("/api/session/export.csv")
        assert r.status_code == 200
        assert "action" in r.text and "STABILIZE" in r.text
