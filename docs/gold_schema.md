# Gold Katmanı — Şema Notu

`python -m src.gold.build_gold` çalıştırıldığında `data/gold/lakehouse.duckdb`
içine 4 tablo yazılır. Lakehouse Tool bu tabloları sorgulayacak.

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

Not: terminalde/log çıktısında yine de `�` görebilirsiniz (ör. `İSTANBUL`
konsolda `�STANBUL` gibi görünür) - bu SADECE konsolun kod sayfası
sorunu, `repr()` / hex codepoint ile kontrol edilince veri doğru
(`0x130` = `İ`). Konsol çıktısına bakıp veri bozuk sanmayın; asıl kontrol
dosyanın/DB'nin byte'larında.

## Tablolar

### `gold_fact_long`
Tüm kaynaklardan (EVDS, TUIK, BDDK Aylık, BDDK Haftalık, BDDK FinTürk-
ulusal) ortak aylık eksende tek satır = tek gözlem şeklinde uzun format.
Genel amaçlı, doğal-dil→SQL tool'unun asıl hedefleneceği tablo.

| Kolon | Açıklama |
|---|---|
| `Period` | `YYYY-MM` |
| `Bolge` | `TR` (ülke geneli) - il bazlı veri için `gold_finturk_regional_quarterly`'e bkz. |
| `Kaynak` | `evds` / `tuik` / `bddk_monthly` / `bddk_weekly` / `bddk_finturk` |
| `Seri_Kod` | Kaynağa özgü seri/tablo kodu (ör. `TP.KFE.TR`, `tablo_4.konut_kredisi`) |
| `Seri_Adi` | İnsan-okunur açıklama (LLM'e prompt'ta bu isim verilecek) |
| `Deger` | Sayısal değer |
| `Birim` | `%`, `milyon TL`, `TRY`, `Bin TL (dogrulanmadi)`, boş/`NULL`=endeks ya da birim teyit edilemedi |
| `Tip` | `stock` (dönem sonu seviye) / `flow` (aya ait akış) / `flow_quarter_ffill` (çeyrek sonundan ay'a forward-fill edilmiş, gerçek aylık gözlem değil) |
| `Kaynak_Dosya` | İzlenebilirlik: hangi silver dosyasından geldi |

**15 seri, 990 satır** (2021-01 → 2026-06), 15/15 seri tam dolu
(66/66 dönem, sıfır null - `gold_fact_long` üzerinde her build'de
kontrol ediliyor). Kaynaklara göre:

- **EVDS (11 seri):** konut fiyat endeksi, konut kredisi faiz oranı,
  TCMB politika faizi, TÜFE, sanayi üretim endeksi, USD/TRY,
  TL mevduat faiz oranı, kredi/banka kartı harcaması (haftalıktan
  aylığa ortalanmış), reel kesim güven endeksi, **M2 para arzı**
  (yeni, haftalıktan aylığa), **birleşik tüketici kredisi faiz oranı**
  (yeni, ihtiyaç+taşıt+konut birlikte). Hepsi `--discover` ile gerçek
  verilerden bulundu, kod uydurulmadı (bkz. `config/evds_series.py`
  yorumları).
- **TUIK (1 seri):** işsizlik oranı.
- **BDDK Aylık (1 seri):** konut kredisi (stok, tablo_4).
- **BDDK Haftalık (1 seri):** konut kredisi haftalık yeni kullandırım
  (akış, tablo_289 - aşağıya bkz., aylığa toplanarak).
- **BDDK FinTürk (1 seri):** toplam nakdi kredi, Türkiye geneli
  (82 il/yurtdışı toplamı, çeyreklikten aylığa forward-fill).

### `gold_bddk_net_profit_monthly`
Faz A2'nin somut çıktısı: BDDK Aylık `tablo_2` içindeki "DÖNEM NET KARI
(ZARARI)" satırı, gerçek veriden **kümülatif olduğu tespit edilip**
(`src/gold/scan_cumulative.py`, bkz. `cumulative_scan_report.csv`)
aylık akışa çevrilmiş hali. `Value_Cumulative` = orijinal YTD değer,
`Value_Monthly` = o aya ait gerçek kâr/zarar.

### `gold_konut_demo`
Hackathon brifinginin kendi örnek senaryosunu (konut kredisi + faiz +
fiyat endeksi + enflasyondan arındırma) uçtan uca üreten geniş tablo.
`Konut_Kredisi_Reel_Milyon_TL` = TÜFE ile 2021-01 bazına deflate
edilmiş. **Birim `milyon TL`** (bkz. aşağıdaki "Birim düzeltmesi" notu
- sütun adı da buna göre `_Milyon_TL` olarak düzeltildi).

**Gerçek veriden çıkan dikkat çekici bulgu:** nominal konut kredisi
2021-01→2026-06 arası ~2.9x büyümüş (276.785 → 801.438 milyon TL), ama
enflasyondan arındırılmış hali aynı dönemde ~3'te 1'e küçülmüş
(276.785 → 99.416 "2021-01 fiyatlarıyla" milyon TL). Bu, Causality
Tool'un vitrin örneği için hazır, gerçek bir hikaye.

### `gold_finturk_regional_quarterly` (yeni)
BDDK FinTürk `tablo_1` (Krediler), **il bazında**, kendi doğal
çeyreklik çözünürlüğünde tutuluyor - 81 il × yurt dışı, aylığa
zorlanmadı (81 il × aylık forward-fill, gerçek bilgi katmadan satır
sayısını 3 katına çıkarırdı). Gold katmanındaki ilk bölgesel (`Bolge`)
boyut. Kolonlar: `Period` (`YYYY-Q`, Q∈{3,6,9,12}), `Bolge` (il adı),
`Toplam_Nakdi_Kredi_Bin_TL`, `Takipteki_Alacak_Bin_TL`,
`NPL_Orani_Yaklasik`. 1804 satır (82 bölge × 22 çeyrek). Ulusal toplamla
çapraz kontrol edildi: `gold_fact_long`'daki `tablo_1.toplam_nakdi_kredi_ulusal`
== bu tablonun 82 ilin toplamı (2021-Q1 için doğrulandı, tam eşleşiyor).

## Çözülen bulgu — haftalık "Konut" satırı stok değil, akış

BDDK Haftalık bülten `tablo_289` ("Loans") / "a) Konut" satırı ile BDDK
Aylık `tablo_4` / "Tüketici Kredileri - Konut" satırı aynı ay için
uyuşmuyordu. Sebebi bulundu: haftalık rakam hafta hafta düz/dalgalı
seyrediyor (38091 → 37319 → 37621 → 38075 → 39452 ...), aylık stok gibi
düzgün büyümüyor - bu, bir bakiye değil, **o hafta açılan yeni konut
kredisi** (akış) imzası. `build_gold.py`'da `Tip='flow'` olarak
düzeltildi ve etiketi netleştirildi ("Haftalik Yeni Kullandirim Toplami
(Bilgi: stok DEGIL)"). Kesin resmi tanım BDDK'nın tablo
dokümantasyonundan/Slack'ten teyit edilebilir ama davranışsal kanıt
güçlü.

## Birim düzeltmesi — BDDK Aylık'ın gerçek isimleri ve birimleri bulundu

`config/settings.py`'ye `BDDK_MONTHLY_TABLES` eklendi: BDDK Aylık'ın 17
tablosunun resmi adı ve birimi, **uydurulmadan** doğrudan bronze
JSON'ın kendi `Json.caption` alanından çıkarıldı (ör.
`data/bronze/bddk_monthly/2021/01/tablo_4_p1.json` →
`"Tuketici Kredileri (milyon TL), Donem:2021/1"`). Bu, tablo_1-17'nin
artık sadece numarayla değil gerçek isimleriyle anılabileceği anlamına
geliyor (Bilanço, Kar Zarar, Krediler, Tüketici Kredileri, Sektörel
Kredi Dağılımı, KOBİ Kredileri, Sendikasyon Seküritizasyon Kredileri,
Menkul Kıymetler, Mevduat Türler İtibarıyla, Mevduat Vade İtibarıyla,
Likidite Durumu, Sermaye Yeterliliği, Yabancı Para Pozisyonu, Bilanço
Dışı İşlemler, Rasyolar, Diğer Bilgiler, Yurt Dışı Şube Rasyoları).

**Bunu yaparken gerçek bir hata bulundu ve düzeltildi:** `tablo_4`'ün
gerçek birimi **"milyon TL"**, önceki gold kodunda yanlışlıkla "Bin TL"
etiketlenmişti. `gold_fact_long`'daki `Deger` sayısı hiç değişmedi,
sadece `Birim` etiketi ve `gold_konut_demo`'daki sütun adları
(`Konut_Kredisi_Bin_TL` → `Konut_Kredisi_Milyon_TL`) düzeltildi. Yani
276.785 rakamı her zaman doğruydu, ama "276.785 bin TL" demek yanlıştı
- doğrusu "276.785 milyon TL" (~277 milyar TL).

**Not:** birim doğrulaması sadece BDDK Aylık için mümkün oldu (caption
alanı sadece orada var). BDDK Haftalık'ın HTML sayfasında hiçbir yerde
birim yazmıyor (kontrol edildi) - `tablo_289.konut`'un `Birim`'i bu
yüzden `NULL` bırakıldı, "Bin TL" gibi bir şey iddia edilmedi. BDDK
FinTürk'ün JSON'ında da caption yok; `tablo_1.toplam_nakdi_kredi_ulusal`
ve `gold_finturk_regional_quarterly`'nin "Bin TL" etiketi
`config/settings.py`'deki `BDDK_FINTURK_TABLES` sözlüğünden miras
kalan, **doğrulanmamış** bir varsayım - `gold_fact_long`'da
`"Bin TL (dogrulanmadi)"` olarak işaretlendi.

## Bilinçli olarak dışarıda bırakılan veri — altın fiyatı (EVDS)

`TP.ALTINPIYASA.KAP05` (Borsa İstanbul altın kapanış fiyatı, TL/gr)
toplandı ve silver'a işlendi ama **gold'a dahil edilmedi**: sadece 207
düzensiz gözlem (28 güne varan boşluklar), ve **2026'da hiç veri yok**
(son gözlem 2025-11-24). Bunu aylığa zorlamak (forward-fill) veri
uydurmak olurdu. Hackathon brifinginin kendi demo senaryosu zaten bu
ihtiyacı Web URL Agent Tool ile Borsa İstanbul'un kendi sayfasından
canlı çekmeyi öngörüyor (slayt 13) - daha doğru kaynak o.

## Bilinen sınırlama — kümülatif tespiti oran (%) tablolarında

`scan_cumulative.py`, `tablo_15` (rasyo/yüzde tablosu) içindeki 9
satırı da "kümülatif" olarak işaretledi, ama bunlar ortalama bazlı
oranlar - basit fark alma (`decumulate()`) bir yüzdeyi anlamlı bir
aylık yüzdeye çevirmez. Bu yüzden `build_gold.py` şu an sadece
`tablo_2` (mutlak TL tutarları) üzerinde decumulate çalıştırıyor;
rasyo tabloları bilinçli olarak dışarıda bırakıldı.

## Sıradaki adımlar (Faz A'nın kalanı)

1. Diğer 16 BDDK Aylık tablosundan (artık hepsinin gerçek adı/birimi
   `BDDK_MONTHLY_TABLES`'da var) + BDDK Haftalık'tan ihtiyaç duyulan
   başka satırları `gold_fact_long`'a ekle (şema hazır, satır eklemek
   ucuz - bkz. `build_bddk_monthly_konut_facts()` deseni).
2. FinTürk'ün diğer 6 tablosu (Mevduat, Bireysel Bankacılık, Seçilmiş
   Sektörel Krediler, Rasyolar, Şube/Nüfus, Altın) henüz eklenmedi -
   sadece tablo_1 (Krediler) işlendi.
3. Yeni EVDS/TUIK seri eklemek için: ASLA kod uydurma - önce
   `python -m src.collectors.evds --discover --search <kelime>` (ya da
   TUIK için ilgili discover akışı) ile gerçek kodu bul,
   `client.get_series(datagroup_code)` ile SERIE_CODE'u doğrula, sonra
   `config/evds_series.py`'ye ekle. Bugün böyle 3 yeni seri eklendi.
