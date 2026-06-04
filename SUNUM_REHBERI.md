# 🎤 SUNUM REHBERİ — Jüri Sunumu Senaryosu

> Yiğit için adım adım demo akışı. Sunum günü bu dosyadan ilerle.
> Konu: **Agentic AI – IoT Entegrasyonu** · Tahmini süre: **8–10 dk**

---

## 0) SUNUM ÖNCESİ HAZIRLIK (5 dk, sınıfa girmeden)

- [ ] Telefonda **hotspot** aç → PC'yi telefonun hotspot'una bağla (okul Wi-Fi'ına GÜVENME).
- [ ] PC'de IP'yi öğren: `hostname -I | awk '{print $1}'`
- [ ] IP değiştiyse sertifikayı yenile (bkz. PROGRESS.md), sonra sunucuyu başlat.
  ```bash
  cd ~/drone-mcp-project
  # YENİ: artık tek komut yeterli — cert.pem/key.pem varsa HTTPS otomatik açılır
  # (telefon kamerası için ZORUNLU). drone-server da aynısını yapar.
  uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
  ```
  > Doğrulama: başlatınca log'da `TLS certs found → starting with HTTPS` görmelisin.
- [ ] **TEK KOMUTLA GO/NO-GO** (sunucu gerekmez, sınıfa girmeden çalıştır):
  ```bash
  uv run python scripts/preflight.py
  ```
  > Ollama + model, sertifika+IP eşleşmesi, kural+agent kararları, 66 test — hepsini kontrol eder.
  > **GO — SİSTEM SUNUMA HAZIR** görürsen tamamsın. Sunucu çalışırken ayrıca:
  > `python scripts/verify_e2e.py https://localhost:8000` (8/8 = bağlantı+kamera+kararlar).
- [ ] Ollama çalışıyor (servis olarak otomatik). Model: **llava:7b** (varsayılan, gerçek vision çalışıyor).
      Kontrol: `ollama list` → llava:7b görünmeli. Yoksa: `ollama pull llava:7b`.
      **ÖNERİ:** Engel tespiti/navigasyon için `qwen2.5vl:7b` çok daha güçlü — `ollama pull qwen2.5vl:7b`,
      sonra dashboard'da MODEL seçicisinden geç. (`llama3.2-vision` mllama mimarisi — bu sürümde sorunlu.)
- [ ] **GÜVENLİK VALFİ:** GPU sıkışırsa (8GB tight) dashboard'da **👁 GÖRÜŞ** butonuna bas → agent
      sadece-telemetri moduna geçer, ~1.5sn'de doğru karar verir. Vision olmadan da stabilizasyon kusursuz.
- [ ] PC'de 3 sekme aç: `/sunum`, `/` (dashboard), ve terminal.
- [ ] Telefonda **Firefox**'ta `https://<ip>:8000/drone` aç, sertifikayı kabul et, kamerayı test et.
- [ ] Dashboard'da telefonun görüntüsü + eğim değeri akıyor mu, **bir kez prova et**.

---

## ⌨ KLAVYE KISAYOLLARI (dashboard'da — sunumda hayat kurtarır)

> Dashboard'da herhangi bir yere tıkla, sonra tuşa bas. `?` ile tam liste açılır.

| Tuş | İşlev | | Tuş | İşlev |
|-----|-------|-|-----|-------|
| `1` | Geleneksel mod | | `Q` | Sakin senaryo |
| `2` | Agentik mod | | `W` | Kademeli eğim |
| `3` | Hibrit mod | | `E` | Kritik tehlike |
| `N` | Navigasyon paneli | | `R` | Engel (kural kör) |
| `P` | **Sunum akışı** (adım adım) | | `X` | Senaryoyu durdur |
| `B` | Hızlı benchmark | | `A` | Görsel analiz |
| `M` | Oturum raporu | | `V` | Vision aç/kapat |
| `T` | Eşik düzenleyici | | `S` | Ses uyarıları |
| `F` | Tam ekran | | `?` | Yardım / kısayollar |

**En önemli:** `F` (tam ekran) → `P` (sunum akışını başlat) → ileri tuşlarıyla ilerle. Her şey ekranda yazıyor.

---

## 1) GİRİŞ — Projeyi tanıt (1 dk)

> "Merhaba, ben Yiğit. Projemin konusu **Agentic AI ile IoT entegrasyonu**.
> Sorduğum soru şu: bir IoT cihazını — burada akıllı telefonu bir drone gibi —
> **geleneksel kurallarla mı** yoksa **uçta çalışan bir yapay zekâ ajanıyla mı**
> kontrol etmeliyiz? İkisini aynı donanımda kurup karşılaştırdım."

Aç: **`/sunum`** (tam ekran, F11). Kapak slaytını göster.

---

## 2) PROBLEM — Neden bu proje? (1.5 dk)

Sunumda **Problem** ve **IoT Entegrasyon Katmanı** slaytlarını göster. 3 problemi anlat:

1. **IF/THEN katılığı:** Geleneksel otomasyon sadece önceden yazılan kuralları bilir; kamerayı "anlamaz", öngörülmeyen durumda kör kalır.
2. **N×M entegrasyon çıkmazı:** Her sensör-model çiftine ayrı kod → bakımı imkânsız. **MCP** bunu evrensel araç arayüzüne indirger (N+M).
3. **Gizlilik:** Bulut LLM'i kamera görüntünü dışarı gönderir. Ben modeli **uçta/yerel** çalıştırıyorum — veri cihazdan çıkmaz.

> "Yani bir IoT zinciri kuruyorum: **algıla → ilet → soyutla → karar ver → eyle** —
> ama karar katmanını agentik bir yapay zekâyla değiştiriyorum."

---

## 3) MİMARİ — Sistem nasıl çalışıyor? (1.5 dk)

Sunumda **akış diyagramı** slaytını göster, parmağınla takip et:

```
Telefon (IoT uç düğümü)  →  WebSocket  →  FastMCP  →  Pydantic-AI + Ollama  →  motor komutu
   kamera + IMU              telemetri      @resource/@tool    yerel karar (edge AI)
```

> "Telefon drone'un gövdesi: kamera gözü, ivmeölçer iç kulağı. Veriyi WebSocket'le
> PC'ye akıtıyor. FastMCP sensörleri `resource`, motorları `tool` olarak sarıyor.
> Beyin ise Ollama'daki yerel görüntü modeli."

---

## 4) CANLI DEMO — En önemli kısım (3 dk)

Dashboard'u (`/`) projeksiyona ver, telefonu eline al.

**a) Telefon = drone gövdesi:**
> "Bu telefon şu an drone. Bakın — dashboard'da kamerası ve **eğim açısı** canlı akıyor."
- Telefonu **eğ** → dashboard'daki **Eğim Açısı** banner'ı yeşil→sarı→kırmızı değişiyor.
- "Bu gerçek sensör; uydurma değil. 30°'yi geçince sistem **STABILIZE** kararı veriyor."

**b) Geleneksel mod (KURAL):**
- Üstten **GELENEKSEL (IF/THEN)** moduna geç.
- Sağdaki **Aktif Karar** kartını göster: action=STABILIZE, RİSK=HIGH, güven %100, gecikme `<1 ms`.
- "Çok hızlı — mikrosaniyeler. Ama sadece eğim sayısına bakıyor, kamerayı anlamıyor."

**c) Agentik mod (LLM):**
- **AGENTİK (LLM)** moduna geç.
- **Aktif Karar** kartında artık ajanın **gerekçesi (reasoning)** yazıyor, `👁 görüş` etiketi ve gerçek **gecikme (ms)** görünüyor.
- "Aynı durumu yapay zekâ kamerayı **görerek** yorumladı ve neden o kararı verdiğini açıkladı. Yavaş ama bağlam-farkında ve açıklanabilir."

**d) Canlı karşılaştırma (GERÇEK ÖLÇÜM):**
- Alttaki **Kural ms** vs **Agent ms** sayaçlarını göster.
- Gerçek rakamlar (gradual_tilt, llava:7b, RTX 4060): **Kural ~0 ms · Agent ~3.700 ms**, ikisi de **0 hata**.
- İlginç bulgu: aynı senaryoda kural 15× STABILIZE derken agent 23× STABILIZE dedi — kamerayı görüp daha temkinli davrandı.
- "İşte mühendislik kararı: ani denge için kural (refleks), karmaşık görev planlama için ajan."

---

## 5) FAIL-SAFE — Final vurgusu (1 dk)

> "Peki yapay zekâ beyni çökerse?"
- Üstteki **Ollama** rozetini göster. Ollama'yı kapat (veya zaten kapalıysa): rozet **kırmızı → fallback** olur.
- Agentik modda kalmaya devam et: sistem çökmüyor, **otomatik güvenli kural moduna düşüyor**, karar kartı `[FALLBACK]` gerekçesi gösteriyor.
> "Sistem hiçbir zaman kontrolü kaybetmiyor. Bu, gerçek bir uçuş kontrolcüsünün olmazsa olmazı."

---

## 6) SONUÇ (0.5 dk)

> "Özetle: aynı IoT donanımı, iki beyin. Biri hızı (kural), diğeri zekâyı (ajan) temsil ediyor.
> Ölçümlerim hangisinin nerede kullanılacağını gösteriyor — ve hepsi tamamen yerel,
> gizlilik korunarak, bulut bağımlılığı olmadan çalışıyor."

Sunumun **Sonuç** slaytındaki grafikleri göster.

---

## 🛡️ JÜRİ SORULARI — Hazır cevaplar

| Soru | Cevap |
|------|-------|
| Neden bulut yerine yerel model? | Gizlilik + internet bağımsızlığı + gecikme kontrolü. IoT/robotik için kritik. |
| MCP olmasa ne olurdu? | Her sensör-model çifti için ayrı kod (N×M). MCP standart arayüzle N+M'e indirir. |
| Agentik yavaşsa ne işe yarar? | Hız için değil, **bağlam-farkındalığı** ve öngörülemeyen duruma uyum için. Refleksi kurala, muhakemeyi ajana bırakıyoruz. |
| Gerçek drone'a aktarılır mı? | Evet — telefon yerine Raspberry Pi + kamera; mimari (WebSocket+MCP+ajan) aynı kalır. |
| Eğim nasıl ölçülüyor? | Kameradan DEĞİL — telefonun ivmeölçer/jiroskopundan (IMU). tilt = max(\|pitch\|, \|roll\|). Yerçekimi referansı. |
| Hangi model? | `llama3.2-vision` (11B), Ollama ile yerelde. RTX 4060 8GB'de çalışıyor. |
| Güvenilirlik? | 27 birim testi geçiyor; ajan çökse bile kural-tabanlı fallback devrede. |

---

## ⚠️ DEMO KURTARMA (bir şey ters giderse)

- **Telefon bağlanmadı:** Aynı hotspot'ta mı? IP doğru mu? `https://` mi? → Yedek: PC'de `/drone`'u localhost'ta aç, simülasyon sensörüyle göster.
- **Kamera açılmadı:** Firefox + HTTPS kullan, sertifikayı kabul et.
- **Ollama yok:** Sorun değil — fail-safe demosu olarak KULLAN ("beyin çökünce sistem ayakta kalıyor").
- **Hiçbir şey çalışmazsa:** `uv run python scripts/benchmark.py --model mock` çıktısını ve `/sunum`'u göster; proje mantığı yine tam anlatılır.
</content>
