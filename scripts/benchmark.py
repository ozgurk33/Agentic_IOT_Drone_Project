#!/usr/bin/env python3
"""
Benchmark — Rule-Based vs Agentic AI
====================================
Runs every synthetic flight scenario through BOTH controllers and produces an
apples-to-apples comparison: decision latency (avg/p50/p95), action mix, error
rate and average confidence.

Outputs (under ``benchmarks/results/<timestamp>_*``):
  * ``*_records.jsonl`` — one DecisionRecord per line (raw data)
  * ``*_records.csv``   — same data, spreadsheet-friendly
  * ``*_summary.json``  — aggregate stats per mode and per scenario
and a formatted comparison table on stdout.

Usage:
    uv run python scripts/benchmark.py                  # auto-detect Ollama
    uv run python scripts/benchmark.py --model mock     # force offline mock LLM
    uv run python scripts/benchmark.py --model ollama   # require live Ollama
    uv run python scripts/benchmark.py --vision --repeat 3
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make ``backend`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config                                       # noqa: E402
from backend.agent_brain.agent_core import (                     # noqa: E402
    agent_step, build_agent, check_ollama, hybrid_step,
)
from backend.agent_brain.mock_model import build_mock_model      # noqa: E402
from backend.metrics import DecisionRecord, MetricsCollector, stopwatch, summarise  # noqa: E402
from backend.mcp_server.mcp_server import run_traditional_rules  # noqa: E402
from backend import scenarios as scen                            # noqa: E402
from backend.state import drone_state                            # noqa: E402


_RULE_TO_ACTION = {
    "NO_DATA": "HOLD",
    "EMERGENCY_STABILIZE": "STABILIZE",
    "STABILIZE": "STABILIZE",
    "HOVER": "HOVER",
}


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_traditional(scenario: scen.Scenario, collector: MetricsCollector) -> None:
    for step in scenario.steps:
        drone_state.latest_telemetry = dict(step)
        with stopwatch() as t:
            result = await run_traditional_rules()
        rule = result.get("rule", "HOVER")
        collector.record(DecisionRecord(
            ts=time.time(),
            mode="traditional",
            action=_RULE_TO_ACTION.get(rule, rule),
            latency_ms=t.ms,
            tilt_deg=result.get("tilt_deg"),
            hazard_level=None,
            motor_speed=result.get("motor_speed"),
            confidence=1.0,
            reasoning=f"rule:{rule}",
            used_vision=False,
            ok=True,
            scenario=scenario.name,
        ))


async def run_agentic(scenario: scen.Scenario, agent, collector: MetricsCollector,
                      use_vision: bool) -> None:
    drone_state.telemetry_history.clear()   # trend memory starts fresh per scenario
    for step in scenario.steps:
        drone_state.note_telemetry(dict(step))
        drone_state.latest_frame_b64 = (
            scen.render_frame_b64(step) if (use_vision and step) else None
        )
        await agent_step(agent, scenario=scenario.name, collector=collector)


async def run_hybrid(scenario: scen.Scenario, agent, collector: MetricsCollector,
                     use_vision: bool) -> None:
    drone_state.telemetry_history.clear()
    for step in scenario.steps:
        drone_state.note_telemetry(dict(step))
        drone_state.latest_frame_b64 = (
            scen.render_frame_b64(step) if (use_vision and step) else None
        )
        await hybrid_step(agent, scenario=scenario.name, collector=collector)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_actions(actions: dict) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(actions.items())) or "—"


def print_table(records: list[DecisionRecord]) -> dict:
    summary = summarise(records)
    line = "─" * 78
    print("\n" + line)
    print("  GENEL KARŞILAŞTIRMA — Kural Tabanlı (traditional) vs Agentik (agentic)")
    print(line)
    header = f"  {'MOD':<12}{'KARAR':>7}{'ORT ms':>10}{'p50 ms':>9}{'p95 ms':>9}{'HATA':>6}{'GÜVEN':>7}"
    print(header)
    print("  " + "-" * 74)
    for mode in ("traditional", "agentic", "hybrid"):
        m = summary["by_mode"].get(mode)
        if not m:
            continue
        lat = m["latency_ms"]
        conf = m["avg_confidence"]
        print(f"  {mode:<12}{m['count']:>7}{lat['avg']:>10.2f}{lat['p50']:>9.2f}"
              f"{lat['p95']:>9.2f}{m['errors']:>6}{(f'{conf:.2f}' if conf is not None else '—'):>7}")
    print(line)

    # Speed ratio headline.
    trad = summary["by_mode"].get("traditional", {}).get("latency_ms", {}).get("avg", 0)
    agen = summary["by_mode"].get("agentic", {}).get("latency_ms", {}).get("avg", 0)
    hyb = summary["by_mode"].get("hybrid", {}).get("latency_ms", {}).get("avg", 0)
    if trad and agen:
        ratio = agen / trad
        print(f"  → Agentik mod, kural tabanlı moddan ortalama {ratio:,.0f}x daha yavaş "
              f"({agen:.1f} ms vs {trad:.2f} ms).")
        print("  → Kural tabanlı: hızlı ve öngörülebilir. Agentik: yavaş ama bağlam-farkında.")
    if hyb is not None:
        print(f"  → Hibrit mod: ortalama {hyb:.1f} ms. Tehlikede refleks ANINDA "
              f"stabilize eder (LLM beklenmez);")
        print("    güvenli durumda agent muhakeme eder — kuralın hızı + agentin bağlamı.")
    print(line)

    # Per-mode action mix.
    for mode in ("traditional", "agentic", "hybrid"):
        m = summary["by_mode"].get(mode)
        if m:
            print(f"  {mode:<12} eylem dağılımı: {_fmt_actions(m['actions'])}")
    print(line + "\n")
    return summary


def print_per_scenario(records: list[DecisionRecord]) -> dict:
    by_scenario: dict[str, list[DecisionRecord]] = {}
    for r in records:
        by_scenario.setdefault(r.scenario or "—", []).append(r)
    out: dict = {}
    print("  SENARYO BAZLI ORTALAMA GECİKME (ms)")
    print("  " + "-" * 74)
    print(f"  {'SENARYO':<22}{'traditional':>14}{'agentic':>12}{'hybrid':>12}")
    for name, recs in by_scenario.items():
        s = summarise(recs)
        t = s["by_mode"].get("traditional", {}).get("latency_ms", {}).get("avg", 0)
        a = s["by_mode"].get("agentic", {}).get("latency_ms", {}).get("avg", 0)
        h = s["by_mode"].get("hybrid", {}).get("latency_ms", {}).get("avg", 0)
        out[name] = s
        print(f"  {name:<22}{t:>14.2f}{a:>12.2f}{h:>12.2f}")
    print("  " + "-" * 74 + "\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(records: list[DecisionRecord], summary: dict,
                  per_scenario: dict, out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / stamp

    jsonl_path = base.with_name(f"{stamp}_records.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), default=str) + "\n")

    csv_path = base.with_name(f"{stamp}_records.csv")
    if records:
        fields = list(records[0].to_dict().keys())
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in records:
                w.writerow(r.to_dict())

    summary_path = base.with_name(f"{stamp}_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "overall": summary,
        "per_scenario": per_scenario,
    }, indent=2, default=str), encoding="utf-8")

    print(f"  Çıktılar kaydedildi:\n    {jsonl_path}\n    {csv_path}\n    {summary_path}\n")
    return summary_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def amain(args: argparse.Namespace) -> int:
    # Decide which model to use for the agentic runs.
    model_choice = args.model
    using_ollama = False
    if model_choice in ("auto", "ollama"):
        probe = await check_ollama()
        if probe.get("reachable") and probe.get("model_available"):
            using_ollama = True
        elif model_choice == "ollama":
            print(f"  HATA: Ollama gerekli ama erişilemiyor / model yok: {probe}")
            return 2
        else:
            print(f"  Ollama bulunamadı ({probe.get('error', 'model yok')}) — "
                  f"mock LLM ile çalışılıyor.")

    if using_ollama:
        agent = build_agent()
        model_label = f"ollama:{config.OLLAMA_MODEL}"
    else:
        agent = build_agent(build_mock_model(seed=args.seed))
        model_label = "mock-vision-controller"

    print(f"\n  Model           : {model_label}")
    print(f"  Görüntü (vision): {'açık' if args.vision else 'kapalı'}")
    print(f"  Tekrar          : {args.repeat}")

    selected = (
        [scen.get_scenario(n) for n in args.scenarios]
        if args.scenarios else scen.all_scenarios()
    )
    print(f"  Senaryolar      : {', '.join(s.name for s in selected)}")

    collector = MetricsCollector(maxlen=100_000, file_logging=False)

    drone_state.drone_connected = True
    for rep in range(args.repeat):
        for scenario in selected:
            await run_traditional(scenario, collector)
            await run_agentic(scenario, agent, collector, args.vision)
            await run_hybrid(scenario, agent, collector, args.vision)

    records = [r for r in collector._records]  # full set
    if not records:
        print("  Hiç kayıt üretilmedi.")
        return 1

    summary = print_table(records)
    per_scenario = print_per_scenario(records)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_label,
        "using_ollama": using_ollama,
        "vision": args.vision,
        "repeat": args.repeat,
        "scenarios": [s.name for s in selected],
        "total_steps": len(records),
    }
    write_outputs(records, summary, per_scenario, Path(args.out), meta)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Rule-Based vs Agentic benchmark")
    p.add_argument("--model", choices=["auto", "mock", "ollama"], default="auto",
                   help="Which LLM backend to use for agentic runs (default: auto).")
    p.add_argument("--vision", action="store_true",
                   help="Attach synthetic camera frames to the agent prompts.")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat the whole scenario set N times.")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Subset of scenario names (default: all).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the mock model.")
    p.add_argument("--out", default=str(config.BENCHMARK_DIR),
                   help="Output directory for result files.")
    args = p.parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
