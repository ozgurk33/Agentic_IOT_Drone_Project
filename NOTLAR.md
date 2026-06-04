# Drone MCP Projesi — Geliştirici Notları

> Bu dosya PC yeniden başlatmaları arasında bağlamı korumak için tutulur.
> Claude oturumu her açıldığında bu dosyayı oku.

---

## Proje Durumu: TÜM FAZLAR TAMAMLANDI ✅

Son güncelleme: 2026-06-03

### 2026-06-03 (part 3) — Hibrit kontrolcü + çoklu-kare hafıza + gerçek 3'lü benchmark (Claude, Yiğit yokken)
İki opsiyonel TODO bitirildi, gerçek vision (llava:7b) ile ölçüldü, sunuma işlendi. **27→46 test, ruff temiz.**

**1) Hibrit kontrolcü (refleks + muhakeme — subsumption / Sistem 1-2):**
- `agent_core.py`: `reflex_decision()` (kural = DroneDecision, <1ms), `arbitrate()` saf hakem
  (`reflex-override`/`reflex-only`/`reflex-floor`/`deliberation`), `hybrid_step()` (senkron;
  **tehlikede LLM'i atlar**), `hybrid_controller_loop()` (canlı: refleks 10 Hz + deliberation ayrı
  coroutine, agent refleksi bloklamaz, Ollama çökse refleks taşır). `_log_and_record()` ortak emitter.
- `state.py`: `hybrid_deliberation` önbelleği. `config.py`: `HYBRID_REFLEX_HZ`, `HYBRID_DELIBERATION_TTL_S`.
- `main.py`: `VALID_MODES`=3, `hybrid_loop` lifespan görevi, broadcast'e `deliberation`.
- Dashboard: HİBRİT düğmesi + "Hibrit ms" kutusu + Karar Kartı'nda arbitrasyon kaynağı.

**2) Çoklu-kare kısa hafıza:** `state.telemetry_history` deque + `note_telemetry()`; `summarise_trend()`
prompt'a "TILT_TREND … RISING/FALLING" satırı ekler (`build_prompt`/`decide` artık `history` alır).

**3) GERÇEK 3'lü benchmark (llava:7b, RTX 4060, gradual_tilt, 0 hata):** Kural ~0 ms / Agentik **3918 ms**
/ Hibrit **1361 ms**. Hibrit ~2.9× hızlı AMA aynı kararlar (23 STABILIZE, agentik ile birebir).
Dosyalar: `benchmarks/results/20260603_055126_*`.

**4) Sunum:** Yeni "İki Katmanlı Düşünce" slaytı (15→16), benchmark slaytı 3'lü gerçek rakamlarla,
yol haritası güncellendi (3 paradigma / 46 test). **Yeni testler:** `test_hybrid.py`, `test_memory.py`.

### 2026-06-03 BÜYÜK İLERLEME — Gerçek vision + canlı dashboard (Claude)
**Gerçek agentik AI artık ÇALIŞIYOR (mock değil!):**
- Ollama kuruldu (v0.30.0). `llama3.2-vision` çalışmıyor (`unknown model architecture: 'mllama'`). **`llava:7b` indirildi ve varsayılan yapıldı.**
- Kod düzeltmesi: yerel vision modelleri OpenAI "tools" desteklemiyor → `build_agent` artık **NativeOutput** (Ollama şema-kısıtlı decode) kullanıyor. Mock/test → tool çıktısı. `mock_model.py` prompted/native yolu için TextPart JSON da döndürüyor.
- **Gerçek ölçüm** (gradual_tilt, llava:7b, RTX 4060): Kural ~0 ms / Agent **~3687 ms**, ikisi de 0 hata. Agent 23×, kural 15× STABILIZE. Sunumun "ölçümler" slaytı gerçek rakamlarla güncellendi.
- Fail-safe kanıtlandı: model yüklense de yüklenmese de `agent_step` çökmüyor.
- Çalıştırma: artık ekstra env gerekmez — `uv run uvicorn ...` llava:7b kullanır. Değiştirmek: `OLLAMA_MODEL=qwen2.5vl:7b`.

**Dashboard'a 3 büyük özellik eklendi:**
- **Agent Karar Kartı** (sağ panel): aksiyon + risk + güven çubuğu + gerekçe + gecikme + 👁 vision. Hem kural hem agent modunda (kural log'larına `extra` eklendi). Açıklanabilirliğin vitrini.
- **Ollama durum rozeti** (üst bar): Hazır✓ / model yok / Kapalı→fallback. 5sn'de bir `/api/ollama`.
- **Canlı gecikme karşılaştırması** (alt bar): Kural ms vs Agent ms, 3sn'de bir `/api/metrics/summary`.
- `ws://`→`wss://` HTTPS uyumu düzeltildi.

**Yeni dosyalar:** `PROGRESS.md` (bağlam dosyası, ÖNCE OKU), `SUNUM_REHBERI.md` (demo senaryosu + jüri soruları).

### 2026-06-03 Geliştirme — Agentic AI–IoT çerçevesi + eğim göstergesi (Claude)
Konu "Agentic AI – IoT Entegrasyonu" olarak teyit edildi ve proje bu dile göre güçlendirildi:
- **Sunum:** Kapak yeniden çerçevelendi (badge + meta: "Agentic AI – IoT Entegrasyonu", IoT uç düğümü / edge AI dili). **Yeni slayt eklendi**: "IoT Entegrasyon Katmanı" (Algıla→İlet→Soyutla→Karar→Eyle→Dayanıklılık, 6 kart). Slayt sayısı 14→**15**. `.mono` CSS sınıfı eklendi.
- **Kod:** Uçta hesaplanan **eğim açısı (°)** göstergesi eklendi:
  - `frontend/drone_client/index.html`: telemetri paketine `tilt` alanı + telefonda renk-kodlu "Eğim Açısı" banner'ı (yeşil<15°, sarı 15-30°, kırmızı≥30° — kural eşikleriyle uyumlu). `refreshTelemetryUI` günceller.
  - `frontend/dashboard/index.html`: tele-strip altına renk-kodlu eğim banner'ı (`t-tilt`); `updateTelemetry` `t.tilt`'i kullanır, yoksa `max(|beta|,|gamma|)` ile hesaplar.
- Doğrulama: pytest **27/27 geçti**, tüm rotalar 200, banner'lar ve IoT slaytı canlı.
- **HTTPS demo için:** `cert.pem`/`key.pem` üretildi (IP 192.168.1.5, SAN ile). `.gitignore`'a `*.pem` eklendi. Okul demosunda IP değişeceği için Firefox + HTTPS önerildi (Chrome flag IP'ye bağlı, zahmetli).
- **Ollama:** Yiğit'in makinesinde kurulu DEĞİL. Kurulum: `curl -fsSL https://ollama.com/install.sh | sh` → `ollama pull llama3.2-vision`. Donanım: RTX 4060 8GB VRAM (11B vision sınırda ama çalışır), 15GB RAM, 16 çekirdek. GERÇEK vision testi henüz yapılmadı — jüri öncesi yapılmalı.

### 2026-06-03 Doğrulama (Claude)
Sıfırdan tam kontrol yapıldı, her şey ÇALIŞIR durumda:
- `uv run pytest` → **27/27 test GEÇTİ** (3.5 sn)
- Tüm modüller import oluyor (main, mcp_server, agent_core, mock_model, scenarios, metrics)
- Sunucu temiz açılıyor; tüm rotalar **200 OK** döndürüyor:
  `/`, `/drone`, `/sunum`, `/health`, `/api/ollama`, `/api/metrics/summary`
- `uv run python scripts/benchmark.py --model mock` → çalışıyor, sonuç CSV/JSONL/summary üretiyor
  (agentik ~114 ms vs kural ~0.01 ms; 112 karar; 0 hata)
- Sunum: 14 slayt, video placeholder + N×M tile'ları mevcut
- **Sonuç:** Proje sunuma HAZIR. Ollama kurulumu tek eksik (sadece gerçek vision modu için; mock ile her şey çalışıyor).


### Tamamlanan Fazlar

| Faz | İçerik | Durum |
|-----|--------|-------|
| 1 | `uv` kurulumu, klasör yapısı, `.env` konfigürasyonu | ✅ |
| 2 | Mobil drone istemcisi (`frontend/drone_client/index.html`) — kamera + IMU → WebSocket | ✅ |
| 3 | FastMCP sunucusu + dashboard + IF/THEN kural motoru | ✅ |
| 4 | Pydantic-AI agent + Ollama entegrasyonu + güvenli fallback | ✅ |
| 5 | Benchmark scripti, 27 pytest testi, JSONL metrik loglama | ✅ |
| 6 | 14 slaytlı Türkçe jüri sunumu (`presentation/index.html`) | ✅ |

---

## Sunucuyu Başlatmak

```bash
cd ~/drone-mcp-project

# Sadece sunucu (Ollama olmadan da çalışır — mock fallback):
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Ollama ile tam agentik mod:
ollama serve &
ollama pull llama3.2-vision
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Önemli URL'ler

| URL | Açıklama |
|-----|----------|
| `http://<ip>:8000/` | Dashboard kontrol paneli |
| `http://<ip>:8000/drone` | Mobil drone istemcisi (telefonda aç) |
| `http://<ip>:8000/sunum` | Jüri sunumu (14 slayt, Türkçe) |
| `http://localhost:8000/health` | Sağlık kontrolü |
| `http://localhost:8000/api/metrics/summary` | Canlı metrik özeti |
| `http://localhost:8000/api/ollama` | Ollama durum kontrolü |

---

## Testler ve Benchmark

```bash
# 27 test (hepsi geçiyor):
uv run pytest

# Benchmark (Ollama olmadan, mock model ile):
uv run python scripts/benchmark.py --model mock

# Benchmark (Ollama ile, vision açık, 3 tekrar):
uv run python scripts/benchmark.py --model ollama --vision --repeat 3

# Log analizi:
uv run python scripts/analyze_logs.py logs/decisions.jsonl
```

---

## Kritik Dosyalar

```
backend/
  config.py          — tüm ortam değişkenleri ve eşikler
  main.py            — FastAPI, WebSocket hub, background loop'lar
  state.py           — DroneState singleton (frame_queue maxsize=1)
  metrics.py         — DecisionRecord + MetricsCollector
  scenarios.py       — 5 sentetik senaryo + yapay ufuk görseli
  mcp_server/
    mcp_server.py    — FastMCP: 4 resource, 6 tool, IF/THEN kurallar
  agent_brain/
    agent_core.py    — Pydantic-AI agent, DroneDecision, dispatch, loop
    mock_model.py    — GPU'suz test için FunctionModel
frontend/
  dashboard/index.html    — PC kontrol paneli
  drone_client/index.html — Mobil HTML5 istemci
presentation/index.html   — Jüri sunumu (Türkçe, 14 slayt)
scripts/
  benchmark.py      — Kural tabanlı vs Agentik karşılaştırması
  analyze_logs.py   — JSONL log analiz aracı
```

---

## Tasarım Kararları (Değiştirme Öncesi Oku!)

- **`frame_queue(maxsize=1)`**: Geri basınç mekanizması. Eski kareler atılır, agent her zaman en güncel kareyi görür.
- **`DashboardManager` 10 Hz yayın**: Dashboard 100ms'de bir güncellenir.
- **Ollama erişilemezse**: Agent `safe_fallback_decision()` çağırır, kural tabanlı güvenli eylem uygulanır. Döngü HİÇBİR ZAMAN çökmez.
- **MCP araçlar in-process**: `dispatch_decision()`, MCP tool fonksiyonlarını doğrudan Python fonksiyon çağrısı olarak çağırır (SSE üzerinden değil). Bu hem hızlı hem de güvenlidir.
- **Mock model**: `build_mock_model()` ile Ollama olmadan tüm agent pipeline'ı test edilebilir.

---

## Paket Versiyonları (Kurulu)

- `pydantic-ai`: 1.104.0
- `fastmcp`: 3.3.1
- `fastapi`: mevcut en güncel
- `Python`: 3.11.x

---

## Yapılabilecek İyileştirmeler (Gelecek)

- [ ] Hibrit kontrolcü: refleks (kural) + muhakeme (agent) aynı anda
- [ ] Çoklu kare bağlamı (kısa hafıza, son 3-5 kare)
- [ ] Daha küçük/hızlı model: `qwen2.5` veya `llava-phi3`
- [ ] WebRTC ile daha düşük gecikme (WebSocket yerine)
- [ ] Gerçek mini-drone donanımına aktarım (Raspberry Pi + Python)
- [ ] GPS ve barometre sensör füzyonu

---

## Olası Sorunlar ve Çözümleri

**Sorun**: Ollama bağlanamıyor  
**Çözüm**: `ollama serve` çalışıyor mu? `curl http://localhost:11434/api/tags` ile kontrol et.

**Sorun**: `llama3.2-vision` modeli yok  
**Çözüm**: `ollama pull llama3.2-vision` (büyük model, ~2-4 GB)

**Sorun**: Telefon kamerası açılmıyor  
**Çözüm**: Tarayıcıda kamera iznini kontrol et. HTTPS veya localhost zorunlu.

**Sorun**: Dashboard drone'u görmüyor  
**Çözüm**: `/ws/drone` WebSocket adresini doğru IP ile gir. Güvenlik duvarı portuna dikkat.

**Sorun**: Import hatası `pydantic_ai.models.ollama`  
**Çözüm**: `pydantic-ai >= 1.0.0` gerekiyor. `uv sync` çalıştır.
