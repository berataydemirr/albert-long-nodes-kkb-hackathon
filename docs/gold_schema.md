# Gold Katmanı — Şema Notu

`python -m src.gold.build_gold` çalıştırıldığında `data/gold/lakehouse.duckdb`
içine 3 tablo yazılır. Lakehouse Tool bu tabloları sorgulayacak.

## ⚠️ Kritik: CSV okuma/yazmada her zaman `encoding="utf-8"`

Bu makinede (Windows, Türkçe yerel ayar) `pd.read_csv(path)` /
`df.to_csv(path)` **varsayılan** çağrıları OS'nin yerel kod sayfasını
(cp1254) kullanıyor. Diskteki CSV'ler gerçekte doğru UTF-8 (bronze JSON
byte'ları ile doğrulandı), ama `encoding="utf-8"` verilmeden okunduğunda
"DÖNEM" gibi kelimelerdeki Türkçe harfler sessizce `�` karakterine
dönüşüyor. Bu, yazarın makinesinde fark edilmeyip demo sırasında
jüri önünde bozuk metin olarak ortaya çıkabilecek türden bir hata.
**Bu veriye dokunan her yeni kod (tool'lar dahil) `pd.read_csv(path,
encoding="utf-8")` ve `df.to_csv(path, encoding="utf-8")` kullanmalı.**
Mevcut `src/processing/*.py` ve `src/gold/*.py` script'leri düzeltildi.

## Tablolar

### `gold_fact_long`
Tüm kaynaklardan (EVDS, TUIK, BDDK Aylık, BDDK Haftalık) ortak aylık
eksende tek satır = tek gözlem şeklinde uzun format. Genel amaçlı,
doğal-dil→SQL tool'unun asıl hedefleneceği tablo.

| Kolon | Açıklama |
|---|---|
| `Period` | `YYYY-MM` |
| `Kaynak` | `evds` / `tuik` / `bddk_monthly` / `bddk_weekly` |
| `Seri_Kod` | Kaynağa özgü seri/tablo kodu (ör. `TP.KFE.TR`, `tablo_4.konut_kredisi`) |
| `Seri_Adi` | İnsan-okunur açıklama (LLM'e prompt'ta bu isim verilecek) |
| `Deger` | Sayısal değer |
| `Birim` | `%`, `Bin TL`, `TRY`, boş=endeks |
| `Tip` | `stock` (dönem sonu seviye) / `flow` (aya ait akış) |
| `Kaynak_Dosya` | İzlenebilirlik: hangi silver dosyasından geldi |

9 seri, 594 satır (2021-01 → 2026-06).

### `gold_bddk_net_profit_monthly`
Faz A2'nin somut çıktısı: BDDK Aylık `tablo_2` içindeki "DÖNEM NET KARI
(ZARARI)" satırı, gerçek veriden **kümülatif olduğu tespit edilip**
(`src/gold/scan_cumulative.py`, bkz. `cumulative_scan_report.csv`)
aylık akışa çevrilmiş hali. `Value_Cumulative` = orijinal YTD değer,
`Value_Monthly` = o aya ait gerçek kâr/zarar.

### `gold_konut_demo`
Hackathon brifinginin kendi örnek senaryosunu (konut kredisi + faiz +
fiyat endeksi + enflasyondan arındırma) uçtan uca üreten geniş tablo.
`Konut_Kredisi_Reel_Bin_TL` = TÜFE ile 2021-01 bazına deflate edilmiş.

**Gerçek veriden çıkan dikkat çekici bulgu:** nominal konut kredisi
2021-01→2026-06 arası ~2.9x büyümüş (276.785 → 801.438 bin TL), ama
enflasyondan arındırılmış hali aynı dönemde ~3'te 1'e küçülmüş
(276.785 → 99.416 "2021-01 fiyatlarıyla" bin TL). Bu, Causality
Tool'un vitrin örneği için hazır, gerçek bir hikaye.

## Bilinen açık soru — kapatılmadı, kapatılmış gibi gösterilmedi

BDDK Haftalık bülten `tablo_289` ("Loans") / "a) Konut" satırı ile BDDK
Aylık `tablo_4` / "Tüketici Kredileri - Konut" satırı **aynı ay için
uyuşmuyor** (ör. 2024-01: haftalık 14.515 bin TL vs aylık 436.688 bin
TL — ne 1000 ne 10 kat fark, temiz bir birim dönüşümü değil). İkisi
muhtemelen farklı şeyi ölçüyor (ör. haftalık bültendeki panel farklı
bankaları kapsıyor olabilir, ya da "a) Konut" o hafta açılan yeni
kredi olabilir, stok değil). `gold_fact_long`'da ikisi de ayrı
`Seri_Kod` olarak duruyor, birbirinin yerine geçmiyor - kimse ikisini
karıştırıp yanlışlıkla toplamasın diye bilerek burada not edildi.
Slack'te BDDK'ya sorulabilir ya da tablo dokümantasyonundan
netleştirilebilir.

## Bilinen sınırlama — kümülatif tespiti oran (%) tablolarında

`scan_cumulative.py`, `tablo_15` (rasyo/yüzde tablosu) içindeki 9
satırı da "kümülatif" olarak işaretledi, ama bunlar ortalama bazlı
oranlar - basit fark alma (`decumulate()`) bir yüzdeyi anlamlı bir
aylık yüzdeye çevirmez. Bu yüzden `build_gold.py` şu an sadece
`tablo_2` (mutlak TL tutarları) üzerinde decumulate çalıştırıyor;
rasyo tabloları bilinçli olarak dışarıda bırakıldı.

## Sıradaki adımlar (Faz A'nın kalanı)

1. Diğer 16 BDDK Aylık tablosundan + BDDK Haftalık'tan ihtiyaç
   duyulan başka satırları `gold_fact_long`'a ekle (şu an sadece
   konut kredisi çekildi - şema hazır, satır eklemek ucuz).
2. FinTürk (çeyreklik, il bazlı) henüz `gold_fact_long`'a dahil değil
   - `src/gold/align.upsample_quarterly_to_monthly()` yazıldı ama
     henüz `build_gold.py`'da kullanılmadı.
3. Haftalık/aylık konut uyuşmazlığını netleştir (yukarıya bkz.).
4. Yeni veri ekleme fikirleri için bkz. ekip sohbetindeki not.
