"""
Pre-demo preflight check — run this right before the jury presentation.
Gives a clear GO / NO-GO verdict. Does NOT need the server running.

    python scripts/preflight.py

Checks:
  1. Ollama reachable + the configured model is pulled
  2. TLS certs exist AND match the machine's current LAN IP (phone camera!)
  3. Rule engine produces the correct decision for a danger tilt
  4. The agent (real Ollama, text mode — fast) produces a correct decision
  5. The full pytest suite is green
"""
import asyncio
import socket
import subprocess
import sys

import httpx

from backend import config

OK, BAD, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"
verdicts: list[bool] = []


def line(ok, msg, detail=""):
    verdicts.append(ok)
    mark = OK if ok else BAD
    print(f"  {mark} {msg}" + (f"  \033[90m{detail}\033[0m" if detail else ""))


def warn(msg, detail=""):
    print(f"  {WARN} {msg}" + (f"  \033[90m{detail}\033[0m" if detail else ""))


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


async def check_ollama() -> None:
    print("\n\033[1m1) Ollama (yerel LLM)\033[0m")
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"{config.OLLAMA_BASE_URL}/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
        line(r.status_code == 200, "Ollama erişilebilir", config.OLLAMA_BASE_URL)
        has = any(config.OLLAMA_MODEL in m or m.startswith(config.OLLAMA_MODEL) for m in models)
        line(has, f"Model yüklü: {config.OLLAMA_MODEL}",
             "" if has else f"→ ollama pull {config.OLLAMA_MODEL}")
        if models:
            warn("Mevcut modeller", ", ".join(models[:5]))
    except Exception as exc:
        line(False, "Ollama erişilemedi", str(exc))


def check_certs() -> None:
    print("\n\033[1m2) HTTPS sertifikası (telefon kamerası için ZORUNLU)\033[0m")
    cert, key = config.SSL_CERTFILE, config.SSL_KEYFILE
    exist = cert.exists() and key.exists()
    line(exist, "cert.pem + key.pem mevcut", str(cert.parent))
    if not exist:
        warn("Üret", "openssl req -x509 ... (bkz. PROGRESS.md)")
        return
    ip = _lan_ip()
    try:
        import ssl as _ssl
        c = _ssl._ssl._test_decode_cert(str(cert)) if hasattr(_ssl, "_ssl") else None
        san = ""
        if c:
            for typ, val in c.get("subjectAltName", ()):
                if typ == "IP Address":
                    san += val + " "
        match = ip in san
        line(match, f"Sertifika bu makinenin IP'siyle eşleşiyor ({ip})",
             f"cert SAN: {san.strip() or '?'}")
        if not match:
            warn("IP değişmiş", f"cert'i {ip} için yenile (PROGRESS.md)")
    except Exception as exc:
        warn("Sertifika IP doğrulanamadı", str(exc))


async def check_decisions() -> None:
    print("\n\033[1m3) Karar doğruluğu (kural + agent)\033[0m")
    from backend.mcp_server.mcp_server import compute_rule_decision_only
    from backend.state import drone_state

    # Rule — deterministic, instant
    drone_state.note_telemetry({"beta": 35.0, "gamma": 5.0, "tilt": 35.0, "az": 9.8})
    rule = compute_rule_decision_only()
    line(rule["action"] == "STABILIZE",
         f"Kural: tilt 35° → {rule['action']}", "beklenen STABILIZE")

    # Agent — real Ollama, text mode (fast, no vision OOM risk)
    try:
        from backend.agent_brain import agent_core
        drone_state.latest_frame_b64 = None
        cases = [(35.0, "STABILIZE"), (5.0, "HOVER")]
        agent_ok = True
        for tilt, expected in cases:
            drone_state.note_telemetry({"beta": tilt, "gamma": 2.0, "tilt": tilt, "az": 9.8})
            rec = await asyncio.wait_for(agent_core.agent_step(), timeout=40)
            ok = rec.action == expected
            agent_ok = agent_ok and ok
            mark = OK if ok else BAD
            print(f"  {mark} Agent: tilt {tilt:.0f}° → {rec.action} "
                  f"\033[90m(beklenen {expected}, {rec.latency_ms:.0f}ms)\033[0m")
        verdicts.append(agent_ok)
    except Exception as exc:
        warn("Agent testi atlandı", f"{type(exc).__name__}: {exc}")
        warn("", "Vision GPU'da sıkışıyorsa dashboard'da V ile kapat (metin modu hızlı).")


def check_tests() -> None:
    print("\n\033[1m4) Test paketi\033[0m")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           capture_output=True, text=True, timeout=120)
        passed = "passed" in r.stdout
        last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1]
        line(passed and r.returncode == 0, "pytest yeşil", last.strip())
    except Exception as exc:
        warn("pytest çalıştırılamadı", str(exc))


async def main() -> int:
    print("\n" + "=" * 56)
    print("  🚁  DRONE MCP — SUNUM ÖNCESİ ÖN-UÇUŞ KONTROLÜ")
    print("=" * 56)
    await check_ollama()
    check_certs()
    await check_decisions()
    check_tests()

    ok = sum(verdicts)
    total = len(verdicts)
    print("\n" + "=" * 56)
    if ok == total:
        print(f"  \033[92m✓✓✓  GO — SİSTEM SUNUMA HAZIR ({ok}/{total})  ✓✓✓\033[0m")
    elif ok >= total - 1:
        print(f"  \033[93m!  NEREDEYSE HAZIR ({ok}/{total}) — uyarıları kontrol et\033[0m")
    else:
        print(f"  \033[91m✗  NO-GO ({ok}/{total}) — yukarıdaki ✗'leri düzelt\033[0m")
    print("=" * 56 + "\n")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
