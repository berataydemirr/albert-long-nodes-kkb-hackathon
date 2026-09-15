# Agent mimarisi

## Genel akış

```text
Kullanıcı sorusu
      ↓
Router
      ↓
MIA model + güvenli tool seti
      ↓
Function call
      ↓
Python tool'u
      ↓
Sonuç modele geri döner
      ↓
Gerekirse yeni tool çağrısı
      ↓
Final cevap
```

## Router

Router soruyu sınıflandırır:

- `local_analytics`
- `web_search`
- `web_url`
- `refuse`

Güvenli isteklerde tüm kayıtlı tool'lar modele açılır. Böylece model birden
fazla aracı zincirleyebilir. Tehlikeli SQL veya prompt-injection istekleri
model çağrılmadan reddedilir.

## Agent loop

1. Soru MIA modeline gönderilir.
2. Model bir veya daha fazla function call üretir.
3. Python ilgili tool'u çalıştırır.
4. Tool sonucu modele geri verilir.
5. Model gerekirse başka bir tool çağırır.
6. Tool çağrısı kalmayınca final cevap döner.

## Pipeline aşamaları

Ekip feedback'i "Veri Okuma / Şema Çıkarma / Analiz Üretme / Soru
Cevaplama / Sonuç Sunma" olarak 5 ayrı modül istedi. Bu ayrım kod
tabanında zaten vardı ama örtüktü - burada hangi dosya/fonksiyonun hangi
aşamaya karşılık geldiği açıkça yazılı:

| Aşama | Dosya(lar) | Görev |
|---|---|---|
| **1. Veri Okuma** | `src/collectors/*.py` (BDDK, EVDS, TUIK - batch); `src/agent/data_import.py::_read_table` (CSV/Excel, runtime); `src/agent/pdf_import.py::_check_pdf_open` + pdfplumber (PDF, runtime); `src/agent/bist_import.py::fetch_bist_stock` (BIST, canlı) | Ham veriyi kaynağından (dosya, API, web) okur. Batch kaynaklar bronze/silver katmanından geçer; runtime kaynaklar (upload, canlı çekim) doğrudan okunur. |
| **2. Şema Çıkarma** | `src/agent/data_import.py::detect_schema`, `_infer_column`, `_raw_header`; `src/agent/pdf_import.py::_table_to_dataframe` | Kolon tiplerini (numeric/text/date_like_text/numeric_like_text), null sayılarını, tekrarlı/boş başlıkları tespit eder - CSV/Excel/PDF için aynı çıktı şeklini üretir, format farkı burada kaybolur. |
| **3. Analiz Üretme** | `src/agent/tools.py` (anomaly/change/correlation/causality/threshold), `src/agent/regional.py` (compare_regions), `src/agent/series_resolver.py` | Seriyi lakehouse'tan okuyup istatistiksel hesaplamayı (Python, modelin kendisi değil) yapar; her sonuç kaynak/dönem/birim bilgisiyle döner. |
| **4. Soru Cevaplama** | `src/agent/orchestrator.py::ask`, `src/agent/router.py` | Router isteği sınıflandırır, model hangi tool'ların hangi sırayla çağrılacağına karar verir, tool sonuçlarını okuyup bir sonraki adımı planlar. |
| **5. Sonuç Sunma** | `web_app.py` (`/api/ask`, `/api/upload`), `static/design/index.html`, `src/agent/charts.py` | Final cevabı, tool trace'ini, süre bilgisini ve (varsa) grafiği kullanıcıya gösterir. |

Bu ayrım gerçek bağımsız modüller anlamına gelmiyor - hepsi aynı Python
process'inde, `ask()` fonksiyonunun tool-calling loop'u üzerinden akıyor.
Feedback'in istediği "birbirinden ayrı geliştirilip sonra birleştirilebilir
parçalar" anlamında ayrım: her aşamanın kendi dosyası, kendi test dosyası
(`tests/test_data_import.py`, `tests/test_pdf_import.py`,
`tests/test_agent_tools.py`, ...) ve kendi sorumluluğu var; biri
değişince diğerleri etkilenmez (örn. PDF okuma mantığı değişse Analiz
Üretme katmanı hiç haberdar olmaz - ikisi arasındaki tek temas noktası
lakehouse'taki bir tablo).

## Tool'lar

**Lakehouse (curated + import edilen veri, aynı dosyada):**
- `lakehouse_schema`: Gold tablolarını, seri kataloğunu VE kullanıcı
  tarafından import edilmiş/canlı çekilmiş tabloları listeler.
- `lakehouse_query`: Read-only SQL çalıştırır - gold_* tablolarını VE
  import edilen tabloları aynı sorguda JOIN edebilir.

**Analiz:**
- `anomaly_analysis`, `change_detection`, `correlation_analysis`,
  `causality_analysis`, `threshold_filter_compare`: seri bazlı istatistik.
- `compare_regions`: il bazlı ~35 metrik üzerinden sıralama/karşılaştırma.

**Veri getirme (lakehouse'a yazan):**
- `list_imported_datasets`, `query_imported_dataset`: yüklenen/çekilen
  dataset'leri keşfet ve sorgula.
- `refresh_tuik_series`: kayıtlı bir TUIK serisini canlı günceller,
  `gold_fact_long`'a yazar (`Kaynak=tuik_live`).
- `search_tuik_dataflows`: TUIK'in ~430 dataflow'luk kataloğunda anahtar
  kelimeyle arar (metadata-only, veri çekmez) - kayıtlı olmayan bir seri
  istendiğinde düz ret yerine gerçek adaylar sunar.
- `fetch_bist_stock`: herhangi bir BIST hissesinin güncel fiyat
  geçmişini Yahoo Finance'ten çeker, lakehouse'a yazar.

**Web:**
- `web_search`: güncel web araması yapar.
- `web_url_extract`: HTML, PDF, Excel, CSV, text veya görsel URL
  içeriği çıkarır.

CSV/Excel/PDF import'u ile doğrudan bir tool YOK - bunlar dosya yolu
gerektirdiği için model tarafından çağrılamaz, `POST /api/upload`
endpoint'i üzerinden (kullanıcının dosya seçmesiyle) tetiklenir; sonucu
lakehouse'a yazıldıktan sonra model yukarıdaki tool'larla sorgulayabilir.

Örnek zincir:

```text
web_search
→ web_url_extract
→ lakehouse_query
→ causality_analysis
→ final cevap
```

Browser UI route'u, çağrılan tool'u, argümanları, sonuç önizlemesini ve final
cevabı gösterir.
