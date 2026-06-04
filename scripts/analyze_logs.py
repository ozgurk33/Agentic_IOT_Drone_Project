#!/usr/bin/env python3
"""
Log analyser — summarise a decisions JSONL file
===============================================
Reads a ``*.jsonl`` decision log (written live by the server or by the
benchmark) and prints the same per-mode aggregate the dashboard shows. Handy
for post-flight analysis without re-running anything.

Usage:
    uv run python scripts/analyze_logs.py                 # default logs/decisions.jsonl
    uv run python scripts/analyze_logs.py path/to/file.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config                       # noqa: E402
from backend.metrics import DecisionRecord, summarise  # noqa: E402


def load(path: Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Keep only fields DecisionRecord knows about.
            allowed = DecisionRecord.__dataclass_fields__.keys()
            records.append(DecisionRecord(**{k: v for k, v in d.items() if k in allowed}))
    return records


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(config.DECISION_LOG_FILE)
    if not path.exists():
        print(f"Log dosyası bulunamadı: {path}")
        return 1
    records = load(path)
    if not records:
        print(f"'{path}' içinde geçerli kayıt yok.")
        return 1

    summary = summarise(records)
    print(f"\nKaynak: {path}  ({summary['total']} karar)\n")
    print(f"{'MOD':<14}{'KARAR':>7}{'ORT ms':>10}{'p95 ms':>10}{'HATA':>7}")
    print("-" * 48)
    for mode, m in summary["by_mode"].items():
        lat = m["latency_ms"]
        print(f"{mode:<14}{m['count']:>7}{lat['avg']:>10.2f}{lat['p95']:>10.2f}{m['errors']:>7}")
    print()
    for mode, m in summary["by_mode"].items():
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(m["actions"].items()))
        print(f"  {mode} eylemleri: {mix}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
