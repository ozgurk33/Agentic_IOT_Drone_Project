# 🚁 PROGRESS — Drone MCP Projesi (BURAYI İLK OKU)

> **Amaç:** PC kapanıp açılınca / Claude limiti sıfırlanınca bağlam kaybolmasın.
> Yeni oturumda Claude bu dosyayı + `NOTLAR.md`'yi okuyup kaldığı yerden devam eder.
> Son güncelleme: **2026-06-04** (doğrulama + kritik düzeltmeler)
>
> **2026-06-04 ÖZET:** (1) `main()` artık cert varsa **HTTPS otomatik** (telefon kamerası için).
> (2) **llava few-shot düzeltmesi** — agent artık eşikleri doğru uyguluyor (tilt 5°→HOVER, 35°→STABILIZE, 4/4).
> (3) 👁 Vision aç/kapa güvenlik valfi. (4) Canlı eşik düzenleyici, CSV export, hızlı benchmark, 3 yeni senaryo.
> (5) Modern UI: glassmorphism, gradient aksan, glow, CANLI göstergesi. (6) **54 test geçiyor.**
> Sunum öncesi tek komutla doğrula: `python scripts/verify_e2e.py https://localhost:8000` (8/8 bekle).

---

## 🎯 TEK CÜMLEYLE PROJE
Akıllı telefon = IoT uç düğümü (drone gövdesi). PC = yapay zekâ beyni.
Aynı sensör verisiyle **iki kontrolcüyü** karşılaştırıyoruz:
**Geleneksel (IF/THEN kural)** vs **Agentic AI (Pydantic-AI + Ollama, uçta/yerel)**.
Konu: **Agentic AI – IoT Entegrasyonu**. Bulut LLM YOK, tamamen yerel.

---

## ✅ DURUM: TÜM FAZLAR BİTTİ + EK GELİŞTİRMELER SÜRÜYOR

| Faz | İçerik | Durum |
|-----|--------|-------|
| 1 | uv kurulumu, klasör yapısı, config | ✅ |
| 2 | Mobil drone istemcisi (kamera + IMU → WebSocket) | ✅ |
| 3 | FastMCP sunucusu + dashboard + IF/THEN kural motoru | ✅ |
| 4 | Pydantic-AI agent + Ollama + güvenli fallback | ✅ |
| 5 | Benchmark + 46 pytest + JSONL metrik | ✅ |
| 6 | 16 slaytlık Türkçe jüri sunumu | ✅ |
| EK | **Hibrit kontrolcü** (refleks+muhakeme) + **çoklu-kare hafıza** | ✅ |

**Test durumu:** `uv run pytest` → 46/46 geçiyor. Sunucu açılıyor, tüm rotalar 200.

---

## 🟢 NELER YAPILDI (kronolojik)

### 2026-06-02 — İlk teslim (tüm fazlar)
- 6 fazın tamamı kodlandı, test edildi, sunum hazırlandı. (Detay: NOTLAR.md)

### 2026-06-03 — Oturum: kamera/ağ + IoT çerçevesi + canlı geliştirmeler
1. **Kamera sorunu çözüldü:** `navigator.mediaDevices undefined` hatası = güvenli bağlam gerekiyor.
   - HTTPS için kendinden imzalı sertifika üretildi: `cert.pem` + `key.pem` (IP 192.168.1.5, SAN'lı).
   - `.gitignore`'a `*.pem` eklendi.
   - Okul demosu için: telefon hotspot + **Firefox + HTTPS** önerildi (IP değişince Chrome flag zahmetli).
2. **Konu "Agentic AI – IoT" olarak teyit edildi**, proje bu dile göre güçlendirildi.
3. **Sunum geliştirildi:** kapak yeniden çerçevelendi + **yeni "IoT Entegrasyon Katmanı" slaytı** (deck 14→15). `.mono` CSS sınıfı eklendi.
4. **Canlı eğim açısı (°) göstergesi** eklendi (telefon + dashboard), renk kodlu (yeşil<15°, sarı 15-30°, kırmızı≥30°). Telefon telemetri paketine `tilt` alanı eklendi.
5. **Agent Karar Kartı (explainability panel)** dashboard'a eklendi — bkz. aşağıdaki TODO/DONE.
6. **Ollama durum rozeti + canlı gecikme karşılaştırması** dashboard'a eklendi.
7. `ws://` → `wss://` otomatik geçiş (HTTPS uyumu) düzeltildi.
8. `SUNUM_REHBERI.md` oluşturuldu (demo senaryosu + jüri soruları).
9. **Gerçek vision modeli çalıştırıldı** (llava:7b, NativeOutput). Kural motoruna yapılandırılmış `extra` eklendi (karar kartı her iki modda çalışsın diye). Sunum "ölçümler" slaytı gerçek rakamlarla güncellendi. config/​.env varsayılanı llava:7b. test_agent dead-port'a sabitlendi.

> Bu oturumdaki kod değişikliklerinin tam listesi için commit yok (repo git değil) — dosyalar:
> `frontend/dashboard/index.html`, `frontend/drone_client/index.html`, `presentation/index.html`, `.gitignore`.

---

### 2026-06-03 — Oturum (part 3): Hibrit kontrolcü + çoklu-kare hafıza + gerçek 3'lü benchmark (Claude, Yiğit yokken)
**İki opsiyonel TODO tamamlandı, gerçek vision ile ölçüldü, sunuma işlendi. 27→46 test.**

1. **Hibrit kontrolcü (refleks + muhakeme) — TODO #2 ✅** — "subsumption / Sistem 1-2" mimarisi:
   - `agent_core.py`: `reflex_decision()` (kural motorunu DroneDecision olarak, <1ms, güven 1.0),
     `arbitrate(reflex, deliberate, tilt)` saf hakem fonksiyonu (4 sonuç: `reflex-override` /
     `reflex-only` / `reflex-floor` / `deliberation`), `hybrid_step()` (benchmark/test için senkron;
     **tehlikede LLM'i tamamen atlar** → kural hızı), `hybrid_controller_loop()` (canlı sunucu için:
     hızlı refleks 10 Hz + yavaş deliberation ayrı coroutine'de; agent refleksi ASLA bloklamaz;
     Ollama çökse refleks tek başına taşır = graceful degradation). `_log_and_record()` ortak emitter
     (agent_step de buna refactor edildi — DRY, davranış birebir korundu).
   - `state.py`: `hybrid_deliberation` önbelleği (düz dict, agent importu yok). `config.py`:
     `HYBRID_REFLEX_HZ=10`, `HYBRID_DELIBERATION_TTL_S=5`. `main.py`: `VALID_MODES` 3'lü, `hybrid_loop`
     lifespan görevi, broadcast'e `deliberation` eklendi, REST/WS mod doğrulaması hibrit kabul eder.
   - Dashboard: **HİBRİT (REFLEKS+LLM)** düğmesi (yeşil), "Hibrit ms" gecikme kutusu, Karar Kartı
     hibrit'i + arbitrasyon kaynağını (`reflex-override` vb.) gösterir.
2. **Çoklu-kare kısa hafıza — TODO #3 ✅**:
   - `state.py`: `telemetry_history` deque (maxlen=`AGENT_MEMORY_FRAMES`=5), `note_telemetry()` tek
     giriş noktası (boş/dropout paketleri geçmişe eklenmez). `main.py` ws_drone bunu kullanır.
   - `agent_core.py`: `summarise_trend()` → "TILT_TREND (last N): 5→12→20° — RISING, +15.0°" satırı
     prompt'a girer (`build_prompt`/`decide` artık `history` alır). Agent artık tek kare yerine
     **eğilimi** görür (yükseliyor mu/düşüyor mu).
3. **GERÇEK 3'LÜ BENCHMARK (llava:7b, RTX 4060, gradual_tilt, 24 karar/mod, 0 hata):**
   - Kural: ~0.00 ms / **Agentik: 3918 ms** (p95 4234) / **Hibrit: 1361 ms** (p95 3613).
   - **Hibrit, agentik'ten ~2.9× HIZLI ama AYNI kararları verir** (ikisi de 23 STABILIZE; kural 15).
     Refleks tehlikede (tilt≥30°) LLM'i atlar → ortalama düşer. Hibrit güven 0.78.
   - Not (jüriye dürüst çerçeve): kaydedilen hibrit gecikmesi "güvenli-komuta-kadar-geçen-süre"dir;
     canlı `hybrid_controller_loop`'ta agent ayrı coroutine'de koştuğu için kontrolü hiç bloklamaz,
     yani canlı sistem bundan da hızlıdır. Dosyalar: `benchmarks/results/20260603_055126_*`.
4. **Sunum:** Yeni slayt **"İki Katmanlı Düşünce: Refleks + Muhakeme"** (deck 15→16). Benchmark slaytı
   **3'lü** gerçek rakamlarla yenilendi (bar + tablo + sonuç). Yol haritası: hibrit & hafıza
   "tamamlandı"ya alındı; istatistikler 3 paradigma / 46 test.
5. **Testler:** `tests/test_hybrid.py` (10) + `tests/test_memory.py` (8) eklendi → **46/46 yeşil**, ruff temiz.
   Benchmark `--model mock` ve `--model ollama --vision` ikisi de 3'lü tablo üretir.

## 🟢 GERÇEK VISION MODELİ ÇALIŞIYOR ✅ (2026-06-03)
- Ollama KURULU (v0.30.0). `llama3.2-vision` ÇALIŞMIYOR (`unknown model architecture: 'mllama'` — yeni Ollama eski mimariyi kaldırmış).
- **Çözüm uygulandı:** `llava:7b` indirildi ve **varsayılan model yapıldı** (`config.py`). Gerçek vision inference uçtan uca ÇALIŞIYOR.
- **Önemli kod düzeltmesi:** Yerel vision modelleri OpenAI "tools" API'sini desteklemiyor (yapılandırılmış çıktı için pydantic-ai bunu kullanıyordu → HTTP 400). `build_agent` artık canlı model için **NativeOutput** (Ollama şema-kısıtlı decode) kullanıyor → zayıf modelden bile geçerli JSON. Mock/testler hâlâ tool çıktısı kullanıyor.
- **GERÇEK ÖLÇÜM (gradual_tilt, 24 karar/mod, llava:7b, RTX 4060):**
  - Kural: ~0.00 ms, 0 hata, güven 1.00
  - Agentik: **~3687 ms** (p95 3810), 0 hata, güven 0.70, vision=True
  - Davranış farkı: kural 15× STABILIZE, agent 23× STABILIZE (kamerayı görüp daha temkinli).
- Sonuçlar sunumun "ölçümler" slaytına işlendi. Benchmark dosyaları: `benchmarks/results/20260603_051505_*`.

## 🔴 YAPILMASI GEREKENLER (öncelik sırası)

1. **Sunum video kaydını çek** ve linki Slayt 1'deki placeholder'a yapıştır. (Kalan TEK zorunlu iş — sana bağlı.)
2. ~~Hibrit kontrolcü modu~~ → ✅ **YAPILDI** (2026-06-03 part 3, aşağıya bak).
3. ~~Çoklu-kare kısa hafıza~~ → ✅ **YAPILDI** (2026-06-03 part 3, aşağıya bak).
4. **(Opsiyonel) Daha iyi reasoning için** `qwen2.5vl:7b` dene (llava bazen hazard etiketini NONE veriyor; aksiyon doğru). `OLLAMA_MODEL=qwen2.5vl:7b`.
5. **(Opsiyonel) Tam 3'lü benchmark** tüm senaryolarla: `uv run python scripts/benchmark.py --model ollama --vision`. (Sunumdaki gerçek rakamlar şimdilik gradual_tilt için işlendi.)

---

## 🚀 ÇALIŞTIRMA KOMUTLARI (hızlı referans)

```bash
cd ~/drone-mcp-project

# Düz HTTP (PC'de localhost testi):
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# HTTPS (telefon kamerası için ZORUNLU — IP değişirse cert'i yenile):
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem

# Yeni IP için cert yenileme:
hostname -I | awk '{print $1}'    # IP'yi öğren
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem \
  -subj "/CN=<IP>" -addext "subjectAltName=IP:<IP>"

# Testler + benchmark:
uv run pytest
uv run python scripts/benchmark.py --model mock
```

### URL'ler
| URL | Ne |
|-----|-----|
| `https://<ip>:8000/` | Dashboard (jüriye projeksiyon) |
| `https://<ip>:8000/drone` | Mobil istemci (telefonda Firefox) |
| `https://<ip>:8000/sunum` | Jüri sunumu (15 slayt) |
| `https://localhost:8000/health` | Sağlık |
| `https://localhost:8000/api/ollama` | Ollama durumu |
| `https://localhost:8000/api/metrics/summary` | Canlı metrik |

---

## 🧠 MİMARİ HATIRLATMA (değiştirmeden önce oku)
- `state.py` → `DroneState` singleton, `frame_queue(maxsize=1)` = geri basınç.
- `mcp_server.py` → FastMCP: `@resource` (kamera/IMU), `@tool` (motor/yön), `run_traditional_rules()`.
- `agent_core.py` → Pydantic-AI `Agent`, `DroneDecision` (action/hazard/reasoning/confidence), `dispatch_decision()` MCP araçlarına yönlendirir, `safe_fallback_decision()` Ollama çökerse devreye girer. ASLA çökmez. **Hibrit:** `reflex_decision()` + `arbitrate()` + `hybrid_step()` (senkron, benchmark) + `hybrid_controller_loop()` (canlı, refleks 10 Hz / deliberation ayrı coroutine). **Hafıza:** `summarise_trend()` prompt'a eğim trendi ekler.
- `main.py` → FastAPI, `broadcast_loop()` (10 Hz dashboard yayını, logları `extra` ile taşır), `traditional_rule_loop()`, `agentic_loop()`, `hybrid_loop()`. Mod `drone_state.agent_mode` ile seçilir: `traditional|agentic|hybrid` (`VALID_MODES`).
- Eğim tanımı her yerde aynı: `tilt = max(|beta|, |gamma|)` derece.
- Eşikler `config.py`: `TILT_WARNING_DEG=15`, `TILT_DANGER_DEG=30`.
</content>
