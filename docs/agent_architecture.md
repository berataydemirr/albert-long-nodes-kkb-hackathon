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

Router roadmap'teki C1–C6 niyetlerine göre yalnızca gerekli tool'ları açar.
Karma bir istek birden fazla kabiliyet gerektiriyorsa bu dar setlerin birleşimi
kullanılır. Tehlikeli SQL veya prompt-injection istekleri model çağrılmadan
reddedilir.

## Agent loop

1. Soru MIA modeline gönderilir.
2. Model bir veya daha fazla function call üretir.
3. Python ilgili tool'u çalıştırır.
4. Tool sonucu modele geri verilir.
5. Model gerekirse başka bir tool çağırır.
6. Tool çağrısı kalmayınca final cevap döner.

## Tool'lar

- `lakehouse_schema`: Gold tablolarını ve seri kataloğunu listeler.
- `lakehouse_query`: Read-only SQL çalıştırır.
- `anomaly_analysis`: Aykırı değerleri bulur.
- `change_detection`: Dönemsel değişimleri bulur.
- `correlation_analysis`: İki seri arasındaki korelasyonu ölçer.
- `causality_analysis`: Lag ve Granger testi yapar; kesin nedensellik kanıtı değildir.
- `threshold_filter_compare`: Bir aralıktaki değişimi baseline ile karşılaştırır.
- `web_search`: Güncel web araması yapar.
- `web_url_extract`: HTML, PDF, Excel, CSV, text veya görsel URL içeriği çıkarır.

Örnek zincir:

```text
web_search
→ web_url_extract
→ lakehouse_query
→ causality_analysis
→ final cevap
```

Browser UI `/api/ask/stream` üzerinden gerçek route, model turu ve tool
başlangıç/bitiş olaylarını anlık gösterir. Tamamlandıktan sonra argümanları,
sonuç önizlemesini, süreleri ve final cevabı gösterir; zamanlayıcıyla uydurma
aşama üretmez.
