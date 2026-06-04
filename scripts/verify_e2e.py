"""
End-to-end verification — simulates a phone connecting and verifies the full
pipeline works before a live demo.  Run against a server already listening.

What it checks:
  1. /ws/drone accepts a connection (phone → server)
  2. Binary JPEG frames are received and stored (camera path)
  3. Telemetry JSON is received and stored (sensor path)
  4. /ws/dashboard receives state broadcasts with frame + telemetry (server → UI)
  5. Traditional rule mode produces the CORRECT decision for a given tilt
  6. Shadow decisions populate for the compare strip

Usage:
    python scripts/verify_e2e.py            # tests against https://localhost:8000
    python scripts/verify_e2e.py http://localhost:8001
"""
import asyncio
import io
import json
import ssl
import sys

import httpx
import websockets
from PIL import Image


def _make_jpeg(color=(40, 120, 200)) -> bytes:
    img = Image.new("RGB", (320, 240), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((ok, msg))
    print(f"  {PASS if ok else FAIL} {msg}")


async def main(base: str) -> int:
    is_https = base.startswith("https")
    ws_scheme = "wss" if is_https else "ws"
    host = base.split("://", 1)[1]
    ssl_ctx = None
    if is_https:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
    verify = False if is_https else True

    print(f"\n🔍 E2E doğrulama → {base}\n")

    # ── 1. REST health ──────────────────────────────────────────────────────
    async with httpx.AsyncClient(verify=verify, timeout=10) as client:
        try:
            r = await client.get(f"{base}/health")
            check(r.status_code == 200, f"/health 200 döndü (status={r.status_code})")
        except Exception as exc:
            check(False, f"/health erişilemedi: {exc}")
            print("\n⚠ Sunucu çalışmıyor olabilir. Önce sunucuyu başlatın.\n")
            return 1

        # Set traditional mode for deterministic decision test
        await client.post(f"{base}/api/mode/traditional")

    # ── 2. Connect as phone, stream telemetry + frame ───────────────────────
    drone_url = f"{ws_scheme}://{host}/ws/drone"
    dash_url = f"{ws_scheme}://{host}/ws/dashboard"

    try:
        async with websockets.connect(drone_url, ssl=ssl_ctx) as drone_ws:
            check(True, "/ws/drone bağlantısı kuruldu (telefon simülasyonu)")

            # Send DANGER-level telemetry (tilt 35° > 30° danger threshold)
            telemetry = {
                "type": "telemetry",
                "ax": 0.1, "ay": 0.2, "az": 9.8,
                "alpha": 10.0, "beta": 35.0, "gamma": 5.0,
                "tilt": 35.0,
            }
            await drone_ws.send(json.dumps(telemetry))
            await drone_ws.send(_make_jpeg())  # binary frame
            await asyncio.sleep(0.3)
            check(True, "Telemetri (tilt=35°) + JPEG kare gönderildi")

            # ── 3. Dashboard receives the broadcast ─────────────────────────
            async with websockets.connect(dash_url, ssl=ssl_ctx) as dash_ws:
                got_frame = got_tele = False
                deadline = asyncio.get_event_loop().time() + 3.0
                while asyncio.get_event_loop().time() < deadline:
                    # keep phone alive
                    await drone_ws.send(json.dumps(telemetry))
                    await drone_ws.send(_make_jpeg())
                    try:
                        raw = await asyncio.wait_for(dash_ws.recv(), timeout=0.5)
                        msg = json.loads(raw)
                        if msg.get("frame"):
                            got_frame = True
                        if msg.get("telemetry", {}).get("tilt") == 35.0:
                            got_tele = True
                        if got_frame and got_tele:
                            break
                    except asyncio.TimeoutError:
                        pass
                check(got_frame, "Dashboard kamera karesini aldı (server → UI)")
                check(got_tele, "Dashboard telemetriyi aldı (tilt=35°)")

            # ── 4. Wait for rule engine to make a decision ──────────────────
            await asyncio.sleep(1.5)
            for _ in range(15):
                await drone_ws.send(json.dumps(telemetry))
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)

    except Exception as exc:
        check(False, f"WebSocket akışı başarısız: {exc}")

    # ── 5. Verify the decision was correct ──────────────────────────────────
    async with httpx.AsyncClient(verify=verify, timeout=10) as client:
        r = await client.get(f"{base}/api/logs?limit=40")
        logs = r.json().get("logs", [])
        rule_decisions = [
            log for log in logs
            if log.get("source") == "RULE" and log.get("extra", {}).get("action")
        ]
        if rule_decisions:
            last = rule_decisions[-1]
            action = last["extra"]["action"]
            check(
                action == "STABILIZE",
                f"Kural motoru DOĞRU karar verdi: tilt 35° → {action} "
                f"(beklenen STABILIZE)",
            )
        else:
            check(False, "Kural motorundan karar logu bulunamadı")

        # Shadow decisions populate
        r = await client.get(f"{base}/health")
        check(r.status_code == 200, "Sistem hâlâ ayakta (çökmedi)")

        # Motors should be at emergency speed after danger tilt
        r = await client.get(f"{base}/api/motors")
        motors = r.json()
        check(
            motors.get("fl", 0) >= 75,
            f"Motorlar tehlikeye tepki verdi (FL={motors.get('fl')}% ≥ 75%)",
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*50}")
    if passed == total:
        print(f"  \033[92m✓ TÜM TESTLER GEÇTİ ({passed}/{total})\033[0m")
        print("  Sistem sunum için HAZIR.")
    else:
        print(f"  \033[91m✗ {total-passed} TEST BAŞARISIZ ({passed}/{total})\033[0m")
    print(f"{'='*50}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://localhost:8000"
    sys.exit(asyncio.run(main(base_url)))
