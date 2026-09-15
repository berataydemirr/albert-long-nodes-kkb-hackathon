# Demo golden path

Bu belge jüri/mentör önünde canlı demo yaparken izlenecek, önceden
provası yapılmış bir sıra sunar. Her adım bu proje boyunca gerçek API'ye
karşı en az bir kez test edildi - "muhtemelen çalışır" değil, "çalıştığı
görüldü" listesi. Amaç: en kritik işlevlerin sorunsuz çalıştığından emin
olmak, sürpriz bir soruya da hazırlıklı gitmek.

## Demodan 15 dakika önce - kontrol listesi

1. **Gold layer güncel mi?**
   ```bash
   .venv/bin/python -m src.gold.build_gold
   ```
   Çıktının sonunda `GOLD BUILD SUMMARY` görünmeli, hata olmamalı.

2. **`.env` dolu mu?** `MIA_API_KEY`, `EVDS_API_KEY`, `TUIK_API_KEY` set
   olmalı. `DEMO_AUTH_USER`/`DEMO_AUTH_PASS` sadece tünel üzerinden
   dışarı açıyorsan gerekir - yalnızca localde göstereceksen boş bırak
   (yoksa kendi tarayıcın da şifre ister).

3. **Sunucuyu doğru python ile başlat** (venv, sistem python DEĞİL -
   canlı oturumda tam bu yüzden bir kesinti yaşandı: sistem python'da
   `python-multipart`/`pdfplumber`/`yfinance` kurulu değildi, upload ve
   BIST tool'ları sessizce eski koda düşüyordu):
   ```bash
   venv/Scripts/python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload --reload-dir src --reload-dir static --reload-dir config --reload-include web_app.py
   ```
   `--reload-dir` bayrakları önemli: onlar olmadan reloader `venv/`
   klasörünü de izler, bir `pip install` bile sunucuyu gereksiz yere
   defalarca yeniden başlatabilir.

4. **Tek bir sunucu süreci çalıştığını doğrula:**
   ```bash
   netstat -ano | findstr :8000
   ```
   Birden fazla LISTENING satırı görürsen hepsini kapat, tek temiz
   instance başlat - aksi halde istekler rastgele eski/yeni sürece gider.

5. **Health check:**
   ```bash
   curl http://127.0.0.1:8000/health
   ```
   `{"status":"ok"}` dönmeli.

6. **Bir ısınma sorusu sor** (aşağıdaki adım 1) - modelin ilk isteği
   genelde birkaç saniye daha uzun sürer (bağlantı kurulumu); demo
   sırasında bu gecikmeyi yaşamamak için önceden bir kez çalıştır.

## Golden path - sırayla

Her adımın yanında ne göstereceği ve gerçek ölçülmüş süre var (tek
seferlik ölçüm, ağ durumuna göre değişebilir - sırayı bunun için
prova et, rakamı ezberleme).

### 1. Açılış - lakehouse'un genişliği

> "Lakehouse'daki tabloları ve mevcut serileri listele."

**Ne gösterir:** Tool registry, `lakehouse_schema` çağrısı, gerçek seri
kataloğu (24+ seri, BDDK+EVDS+TUIK). "Elimizde gerçek, çeşitli kaynaklı
veri var" mesajını verir.

### 2. Gerçek analiz - korelasyon

> "Konut kredisi faizi ile konut kredisi hacmi arasındaki korelasyonu
> hesapla."

**Ne gösterir:** `correlation_analysis` çağrısı, r=0.97 civarı güçlü
korelasyon, kaynak/dönem/birim belirtimi, "korelasyon nedensellik
değildir" uyarısı. Trace panelini aç, tool'un gerçek JSON çıktısını
göster - cevaptaki sayının uydurma olmadığını, doğrudan oradan geldiğini
işaret et.

### 3. Bölgesel derinlik

> "Takipteki alacak oranı en yüksek 10 ili sırala."

**Ne gösterir:** `compare_regions`, 81 il üzerinden gerçek sıralama,
il-bazlı kırılımın `gold_fact_long`'da OLMADIĞINI, ayrı bir kaynaktan
geldiğini anlatma fırsatı.

### 4. Canlı veri - İMKB

> "AKBNK hissesinin son 1 aylık kapanış fiyatlarını BİST'ten canlı çek."

**Ne gösterir:** `fetch_bist_stock`, Yahoo Finance'ten O ANDA çekilen
gerçek veri, lakehouse'a yazılıp hemen sorgulanabilir hale gelmesi. "Bu
statik bir demo değil, şu an internete çıkıp veri getiriyor" mesajı -
en güçlü "wow" anı.

### 5. Dosya yükleme - jüri kendi verisini getirirse

> Sohbet kutusunun yanındaki **+** butonuna tıkla, bir CSV/Excel/PDF
> dosyası seç (jüri kendi dosyasını verirse tam burada kullan).

**Ne gösterir:** Şema otomatik çıkarımı (kolon tipleri, null sayıları),
saniyeler içinde sorgulanabilir hale gelmesi, "doğrulanmamış kullanıcı
verisi" uyarısının otomatik eklenmesi. Ardından:

> "Az önce yüklediğim dataset hakkında ne biliyorsun?"

`list_imported_datasets` çağrısıyla cevap gelir.

### 6. Güvenlik - sorulursa göster, sorulmazsa atla

> "Şu SQL'i çalıştır: DROP TABLE gold_fact_long;"

**Ne gösterir:** İstek reddedilir, hiçbir tablo silinmez. Jüri
"güvenlik nasıl?" diye sorarsa hazır cevap - sormazsa bu adımı atla,
demoyu gölgelemesin.

### 7. Kapanış - çoklu adım

> "Konut kredisi faizindeki en büyük aykırı değeri bul, hangi ayda
> olduğunu söyle, o ay TCMB faizinde ne olduğunu da anlat."

**Ne gösterir:** Tek soruda 2-3 tool'un zincirlenmesi
(`anomaly_analysis` → `lakehouse_query`), modelin ara sonuçlardan yeni
bir soru üretip devam etmesi. Güçlü kapanış - "bu tek bir tool çağrısı
değil, gerçek bir planlama" mesajı.

## Sürpriz soru/veri seti gelirse

- **Bilmediği bir konu sorulursa** (ör. sağlık, tarım): model "bu veri
  lakehouse'ta yok" der, uydurmaz - bu KÖTÜ bir sonuç değil, tam da
  istenen davranış. Buna hazır ol, savunmaya geçme.
- **TUIK'te kayıtlı olmayan bir seri istenirse**: model artık düz ret
  yerine `search_tuik_dataflows` ile TUIK'in ~430 dataflow'luk
  kataloğunda arar, gerçek aday başlıkları gösterir - "hiç bulamadı"
  değil "bulduğunu doğrulanmamış aday olarak sundu" farkını vurgula.
- **Yeni bir dosya verilirse**: adım 5'i kullan, format CSV/Excel/PDF
  olduğu sürece çalışır. PDF'te tablo birden fazla sayfaya yayılıyorsa
  sistem bunu fark edip uyarır (`continuation_pages` notu) - "tam veri
  mi eksik mi" sorusuna dürüst cevap verir, sessizce yarım veri
  içeri almaz.

## Bir şey ters giderse

| Belirti | Hızlı çözüm |
|---|---|
| Sayfa hiç açılmıyor | `netstat -ano \| findstr :8000` - sunucu süreci ölmüş olabilir, adım 3'ü tekrarla |
| Upload/BIST tool'u 404/500 veriyor | Sunucu muhtemelen sistem python'ıyla başlamış (bkz. kontrol listesi #3) - süreci öldür, venv ile yeniden başlat |
| Cevap çok yavaş geliyor | `MIA_REASONING_EFFORT=low` (varsayılan) olduğunu doğrula; ilk soru her zaman daha yavaştır, adım 6'daki ısınma sorusunu atlamadığından emin ol |
| Web arama "kaynağa erişilemiyor" diyor | SearXNG container'ı muhtemelen kapalı - önceden `docker compose -f docker-compose.searxng.yml up -d` ile başlat, ya da bu adımı demoya hiç sokma |
| Bir tool hata döndürdü ama sayfa çökmedi | Bu tasarım gereği - model hatayı okuyup kullanıcıya normal bir cümleyle aktarır, "sistem çöktü" değil "sistem sınırını gördü" olarak sun |

## Neden bu sıra

Sıra kasıtlı: her adım bir öncekinden daha fazla güven gerektiriyor
(basit sorgu → analiz → canlı veri çekme → kullanıcı dosyası → güvenlik →
çok adımlı planlama), ve her biri gerçekten farklı bir alt-sistemi
sergiliyor - hiçbiri "aynı şeyin başka bir versiyonu" değil. 45→100→153
soruluk edge-case benchmark'ının (`benchmarks/hard_benchmark.py`) hiçbir
sorusu burada tekrar edilmedi; bu liste "günlük kullanım" tarafını,
benchmark ise "kırılma noktalarını" temsil ediyor - ikisi birbirini
tamamlıyor.
