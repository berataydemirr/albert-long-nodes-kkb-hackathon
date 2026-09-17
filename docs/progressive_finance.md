# Ardışık finans analizi sözleşmesi

## Akış

Türkçe soru → deterministik yetenek router'ı → modelin araç döngüsü →
Python/DuckDB hesabı → değişmez veri kaydı → kaynaklı Türkçe yanıt.
Sonraki soru önceki `turn_id` kayıtlarını taşır. Sunucu gerçek geçmişi
`data/gold/evidence.sqlite3` içinden okur; istemcinin yanıt metnini kanıt
olarak kabul etmez. `recall_turn` eski sorunun tam metnini ve veri kimliklerini,
`evidence_query` kayıtların orijinal satırlarını getirir.

Bu ayrı bir LLM bellek sınıflandırıcısı gerektirmez: router yetenek seçer;
bellek soru, kaynak, araç parametreleri ve sonuç kimliklerini tutar. Web/dosya
içeriği ve istemci özetleri sistem mesajına yükseltilmez. Araçlar bağımsız
LLM ajanları değildir; açıklamaları giriş/çıkış sözleşmesidir. Sistem prompt'u
ortak doğruluk kurallarını, araç açıklamaları kendi parametrelerini tanımlar.

## Veri katmanı

- CSV/Excel yükleme mevcut `/api/upload` üzerinden çalışır. UI varsayılan
  tablo adına içerik sürümü ekletir; aynı adlı farklı dosya eski tabloyu
  ezmez. API'de açıkça `dataset_name` verilmesi mevcut değiştirme
  davranışını korur. Yükleme de sunucunun soru/veri geçmişine kaydedilir.
- `import_url_dataset` doğrudan CSV/Excel URL'sini indirir, tam veriyi
  sürümlü bir DuckDB tablosuna yazar; kaynak URL ve içerik hash'i döner.
  10 MB indirme / 200.000 import satırı sınırları geçerlidir. Kısaltma
  bildirilir; kısaltılmış sonuç korelasyonda tam veri sayılmaz.
- `search_evds_series` gerçek grup/seri metadata'sını keşfeder.
  `fetch_evds_series` seçilen kodu o grupta doğrular; yıllık parçalarla
  indirir; ham veriyi ve aylık dönüşümü ayrı sürümlü tablolarda saklar.
  Ortalama/son/toplam seçimi ve eksik aylar sonuçta belirtilir.
- `web_url_extract` metin/tablo **önizlemesidir**; tam veri ithalatı değildir.
  Genel HTML sayfaları otomatik olarak Gold'a dönüştürülmez.
- DuckDB tablo ve import kataloğu aynı transaction içinde güncellenir.
- Her başarılı JSON araç sonucu prompt kısaltılmadan önce ayrı SQLite
  kanıt deposuna kaydedilir. Gold veya import tablosu sonradan değişse bile
  önceki sorgunun döndürdüğü değerler korunur. Genel sorgu 200 satırla
  sınırlıdır; aylık demo aralığı bu sınırın altındadır.

## Üç soruluk demo

1. “TCMB EVDS bağlantısı üzerinden son iki yılın aylık ihtiyaç kredisi
   faiz oranlarını çek ve listele.”
   Son tamamlanan 24 takvim ayı, doğrulanmış ihtiyaç kredisi serisi,
   aylık gözlemler ve kaynak/birim. Birleşik tüketici/konut kredisi ikame
   edilmez. Güncel veri eksikse tarih aralığı sessizce kaydırılmaz.
2. “Şimdi aynı zaman dilimi için aylık enflasyon verilerini web üzerinden
   sisteme getir.”
   İlk sorunun kesin aralığı korunur. TÜFE endeksi kullanılırsa bir önceki
   ay da alınır. Endeks seviyesi, aylık değişim ve yıllık değişim farklıdır.
3. “İlk soruda bulduğun ihtiyaç kredisi faiz oranları ile az önce çektiğin
   enflasyon verisini birleştir ve aralarındaki korelasyonu hesapla.”
   `evidence_correlation` önceki iki sonucun kimliklerini alır; ay
   tekilliğini, sonlu sayıları ve ortak ayları kontrol eder; DuckDB
   `corr` ile Pearson r hesaplar. TÜFE endeksi girdisinde
   `y_transform=monthly_pct_change` ardışık aylar için
   `100*(endeks/önceki_endeks-1)` üretir. Eksik ay üstünden fark alınmaz.

Pearson r **−1…+1** aralığındadır. Negatif işaret korunur. Sabit seride
sonuç `null` (tanımsız); üçten az ortak ay veya çelişkili aynı-ay değerinde
hesap reddedilir. Sonuç ortak ay sayısını, dışlanan ayları, dönüşümü ve
kaynak kimliklerini içerir. Korelasyon nedensellik kanıtı değildir.

EVDS frekans ve formül ayrımı için resmi kaynak:
[TCMB EVDS Web Servis Kullanım Kılavuzu](https://evds2.tcmb.gov.tr/help/videos/EVDS_Web_Servis_Kullanim_Kilavuzu.pdf).

## Birikme ve sıkıştırma

Üç farklı birikme problemi vardır: model bağlamında büyüyen geçmiş,
aynı dönem için yinelenen veri sürümleri ve ekonomik serinin YTD olması.
Bunlar aynı çözümle düzeltilmez. Değişmez kayıtlar geçmişi korur;
Gold resolver yinelenen dönemleri/karışık birimleri/bölgeleri reddeder;
`decumulate` yalnızca Ocak veya ardışık ay farklarından aylık değer üretir.
Ocak eksikse ilk gözlemi aylık değer diye etiketlemez. YTD tespiti
heuristiktir; faiz/endeks/yüzde üzerinde otomatik uygulanmamalıdır.

Harness her model turundan önce `compact` işlemini değerlendirir.
Tetik: `(tahmini mesaj tokenları + araç şemaları) / (kapasite − çıktı rezervi)
> 0.80`. `AGENT_CONTEXT_TOKENS=16384`, `AGENT_OUTPUT_RESERVE_TOKENS=2048`
dağıtım varsayılanlarıdır, modelin gerçek limitine dair iddia değildir.
Sağlayıcının limitine göre ayarlanmalıdır. Token hesabı yaklaşık karakter
hesabıdır. Eski araç çıktıları kısaltılır; kimlikler korunur; sistem ve
kullanıcı soruları değiştirilmez. Yakın üç araç sonucu korunur.
İşlem ve kazanım API progress/timing alanlarından izlenebilir.

UI yerel saklama v2'ye geçirilir, önceki sohbetler okunabilir. Sunucuda tam
soru/yanıt saklanır; tarayıcı en fazla 100 tur kimliği, görünüm için son
60 mesajı taşır. Model yakın altı turun kısa metnini ve eski turların
indeksini görür; tam geçmiş gerektiğinde `recall_turn` kullanır.
Kimlikler bearer capability niteliğindedir; bu yerel demo, hesap bazında
çok kiracılı veri izolasyonu sağlamaz. Mevcut import tabloları paylaşımlıdır.

## Doğrulama ve mevcut veri

`python -m pytest -q` mevcut regresyonları ve `test_progressive_finance.py`
senaryolarını çalıştırır. Testler deterministik veride r=−1 / +1, sabit seri,
eksik ay, çelişkili dönem, yeniden yükleme sonrası eski değerlerin korunması,
80% eşiği, salt okunur sorgu ve talimat enjeksiyonu sınırını denetler.
Bunlar canlı model doğruluğu veya mentörlerin görülmemiş soruları için
garanti değildir. Kabul toleransı veri/dönüşümle birlikte tanımlanmalıdır;
katsayı için test toleransı `1e-6` uygundur, yalnızca yuvarlama esnetilir.

17 Eylül 2026 yerel incelemesi: Gold'da 24 seri, 1.584 satır,
2021-01…2026-06; boş değer veya yinelenen seri/ay bulunmadı. İhtiyaç
kredisine en yakın mevcut seri ihtiyaç+taşıt+konut birleşimidir. Saf
ihtiyaç kredisi için canlı keşif gerekir. Bu ortamda `EVDS_API_KEY`
tanımlı olmadığı için canlı EVDS uçtan uca doğrulaması yapılamadı.
