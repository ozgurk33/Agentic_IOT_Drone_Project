"""Tests for the metrics collector & summariser."""
from backend.metrics import DecisionRecord, MetricsCollector, _percentile, summarise


def _rec(mode, action, latency, **kw):
    return DecisionRecord(ts=0.0, mode=mode, action=action, latency_ms=latency, **kw)


def test_percentile_basic():
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 50) == 30
    assert _percentile([], 50) == 0.0
    assert _percentile([7], 95) == 7


def test_collector_ring_buffer_caps():
    c = MetricsCollector(maxlen=3, file_logging=False)
    for i in range(5):
        c.record(_rec("agentic", "HOVER", float(i)))
    recent = c.recent(10)
    assert len(recent) == 3
    assert [r["latency_ms"] for r in recent] == [2.0, 3.0, 4.0]


def test_summary_splits_by_mode():
    recs = [
        _rec("traditional", "HOVER", 0.5, confidence=1.0),
        _rec("traditional", "STABILIZE", 0.7, confidence=1.0),
        _rec("agentic", "HOVER", 100.0, confidence=0.8),
        _rec("agentic", "STABILIZE", 200.0, confidence=0.9, ok=False),
    ]
    s = summarise(recs)
    assert s["total"] == 4
    assert s["by_mode"]["traditional"]["count"] == 2
    assert s["by_mode"]["agentic"]["errors"] == 1
    assert s["by_mode"]["agentic"]["actions"] == {"HOVER": 1, "STABILIZE": 1}
    assert s["by_mode"]["agentic"]["avg_confidence"] == 0.85


def test_jsonl_written(tmp_path):
    path = tmp_path / "out.jsonl"
    c = MetricsCollector(jsonl_path=str(path), file_logging=True)
    c.record(_rec("agentic", "HOVER", 1.0))
    c.record(_rec("traditional", "STABILIZE", 0.2))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["action"] == "HOVER"
