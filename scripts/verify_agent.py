"""Verify the AGENTIC + NAV paths with the real Ollama vision model."""
import asyncio
import io
import json
import ssl
import sys

import httpx
import websockets
from PIL import Image

BASE = "https://localhost:8000"
WSB = "wss://localhost:8000"
_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def jpeg(c=(60, 140, 210)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (320, 240), c).save(b, "JPEG", quality=70)
    return b.getvalue()


async def test_agentic() -> bool:
    print("=== AGENTIC karar testi (gerçek llava:7b, ~5-10s) ===")
    async with httpx.AsyncClient(verify=False, timeout=30) as c:
        await c.post(f"{BASE}/api/mode/agentic")
    async with websockets.connect(f"{WSB}/ws/drone", ssl=_ctx) as ws:
        tele = {"type": "telemetry", "ax": 0, "ay": 0, "az": 9.8,
                "alpha": 0, "beta": 35, "gamma": 5, "tilt": 35}
        for _ in range(75):  # 15s of streaming so the 1Hz agent loop fires
            await ws.send(json.dumps(tele))
            await ws.send(jpeg())
            await asyncio.sleep(0.2)
    async with httpx.AsyncClient(verify=False, timeout=10) as c:
        logs = (await c.get(f"{BASE}/api/logs?limit=60")).json()["logs"]
    agent = [x for x in logs if x.get("source") == "AGENT" and x.get("extra", {}).get("action")]
    if agent:
        a = agent[-1]["extra"]
        print(f"  \033[92m✓\033[0m AGENT: action={a['action']} hazard={a.get('hazard')} "
              f"conf={a.get('confidence')} vision={a.get('used_vision')} lat={a.get('latency_ms')}ms")
        print(f"      → {agent[-1]['message'][:110]}")
        return a["action"] in ("STABILIZE", "EMERGENCY_STOP")
    print("  \033[91m✗\033[0m Agent kararı yok. Son loglar:")
    for x in logs[-6:]:
        print("     ", x.get("source"), x.get("message", "")[:80])
    return False


async def test_nav() -> bool:
    print("\n=== NAV görsel analiz testi (gerçek vision, ~5-10s) ===")
    async with websockets.connect(f"{WSB}/ws/drone", ssl=_ctx) as ws:
        await ws.send(json.dumps({"type": "telemetry", "beta": 0, "gamma": 0, "tilt": 0}))
        await ws.send(jpeg((200, 50, 50)))
        await asyncio.sleep(0.5)
        async with httpx.AsyncClient(verify=False, timeout=40) as c:
            d = (await c.post(f"{BASE}/api/nav/analyze")).json()
    mark = "\033[92m✓\033[0m" if d.get("ok") else "\033[91m✗\033[0m"
    print(f"  {mark} NAV: action={d.get('action')} obstacle={d.get('obstacle_detected')} "
          f"vision={d.get('used_vision')} lat={d.get('latency_ms')}ms")
    print(f"      görüyor: {str(d.get('what_i_see'))[:90]}")
    print(f"      kural-kör: {d.get('rule_would_say')}")
    return bool(d.get("ok"))


async def main() -> int:
    try:
        a_ok = await test_agentic()
        n_ok = await test_nav()
    except Exception as exc:
        print(f"  \033[91m✗\033[0m Hata: {exc}")
        return 1
    print(f"\n{'='*50}")
    print(f"  Agentic: {'GEÇTİ ✓' if a_ok else 'BAŞARISIZ ✗'}   "
          f"Nav: {'GEÇTİ ✓' if n_ok else 'BAŞARISIZ ✗'}")
    print(f"{'='*50}\n")
    return 0 if (a_ok and n_ok) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
