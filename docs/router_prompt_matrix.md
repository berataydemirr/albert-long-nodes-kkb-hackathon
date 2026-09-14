# Router prompt matrisi

Router, `01_roadmap_and_tasks.docx` içindeki C1–C6 kullanım alanlarını
deterministik olarak minimum tool setine çevirir. Bir soru gerçekten birden
fazla kabiliyet istiyorsa setlerin birleşimi açılır; güvenli olmak tek başına
bütün tool'ların açılması anlamına gelmez.

| İstek | Route | İzinli tool'lar | Gerekçe |
|---|---|---|---|
| Tablo/kolon/seri kataloğunu listele | `local_analytics` | `lakehouse_schema` | C1 şema keşfi |
| Yerel veri getir veya özetle | `local_analytics` | `lakehouse_schema`, `lakehouse_query` | C1 text-to-SQL |
| Aykırı dönemleri bul | `local_analytics` | C1 + `anomaly_analysis` | C4 anomaly |
| En büyük değişimleri/trend kırılmasını bul | `local_analytics` | C1 + `change_detection` | C6 change detection |
| İki seri arasındaki korelasyon | `local_analytics` | C1 + `correlation_analysis` | C5 korelasyon |
| Lag/Granger/nedensellik | `local_analytics` | C1 + `causality_analysis` | C5 lag + Granger |
| `%x–%y` aralığındaki dönemleri kıyasla | `local_analytics` | C1 + `threshold_filter_compare` | C5 filtre-kıyasla |
| Güncel/web bilgisi | `web_search` | `web_search`, `web_url_extract` | C2 arama + kaynağı doğrulama |
| Verilmiş URL'yi oku | `web_url` | `web_url_extract` | C3 URL çıkarma |
| Güncel veriyi bul ve yerel seriyle kıyasla | `web_search` | C1 + C2 tool'ları | Gerçek karma istek |

`C1` analiz satırlarında `lakehouse_schema` ve `lakehouse_query` anlamına
gelir. Registry sırası korunur; bu sayede prompt ve test çıktıları
deterministiktir.

## Öncelik sırası

1. Boş/geçersiz soru → `refuse`
2. Güvensiz SQL veya talimat aşma isteği → `refuse`
3. URL içeren istek → `web_url`
4. Web/güncel istek → `web_search`
5. Diğer tüm istekler → `local_analytics`

URL, route etiketi bakımından genel web anahtar kelimesinden daha özeldir.
Ancak kullanıcı URL'ye ek olarak açıkça geniş web araması isterse iki web
tool'u da açılır. Domain kelimeleri (örneğin "faiz") tek başına güncel bir
soruya yerel lakehouse tool'larını eklemez; bunun için yerel analiz veya
kıyaslama niyeti gerekir.

## Doğrulama

Route ve exact tool-set beklentileri `tests/test_router_prompt_matrix.py`
içinde kontrol edilir:

```bash
pytest tests/test_router_prompt_matrix.py -q
```
