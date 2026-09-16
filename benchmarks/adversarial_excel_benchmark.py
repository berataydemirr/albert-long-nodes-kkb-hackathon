"""Adversarial benchmark against PRIVATE, real user-uploaded Excel/CSV/PDF
files - not synthetic fixtures, not the public BDDK/TUIK/EVDS gold layer.

The files themselves live in test_excels/ (gitignored - see .gitignore's
"Private real-world test Excel/PDF files" entry), a folder the user drops
new files into over time. This script and its question sets ARE committed;
the files are not, and neither their names nor their content belong in any
committed file, commit message, or README.

Unlike benchmarks/hard_benchmark.py (which checks tool selection and
numeric grounding across ~150 varied questions against the known gold
schema), this benchmark asks a SMALL number of deliberately hard,
trap-laden questions against ONE real file's actual, quirky content -
questions designed to catch a specific class of failure: the model
treating superficially similar columns as interchangeable, or reporting a
plausible-sounding but WRONG fact instead of reading the file's own
ground truth. Each question below states its trap and its ground truth
explicitly, both verified by hand against the real file before being
written down here (see the commit that added this file for how).

Question sets are keyed by a signature of the file's own column names
(see _column_signature), NOT by filename - a real file's name (and this
matters: it can itself be private, e.g. carry a date or an internal
label) never appears anywhere in this committed script, only the shape
of its columns does. This also means a question set survives the same
file being re-sent under a different filename later, which a filename
key would not. A file whose columns match no known signature is still
imported and schema-checked (so nothing in test_excels/ is silently
skipped), just without the hand-crafted traps.

Run with:
    python -m benchmarks.adversarial_excel_benchmark
    python -m benchmarks.adversarial_excel_benchmark --file <stem>
"""
import argparse
import json
import re
import time
from pathlib import Path

from config.settings import PROJECT_ROOT
from src.agent.data_import import detect_schema, import_dataset
from src.agent.orchestrator import ask
from src.agent.pdf_import import import_pdf_table, list_pdf_tables
from src.common.logging_setup import get_logger

logger = get_logger("adversarial_excel_benchmark")

TEST_EXCELS_DIR = PROJECT_ROOT / "test_excels"
REPORT_PATH = Path(__file__).parent / "adversarial_excel_benchmark_report.json"


# --- question sets, one per known file stem -------------------------------
#
# Each question carries:
#   trap        - what specific failure this is fishing for (documentation,
#                 not shown to the model)
#   ground_truth - what was verified by hand against the real file
#   must_include / must_not_include - lowercased substrings the answer is
#                 auto-checked against (best-effort signal, not a hard
#                 grade - see report's own "auto_check" field: a genuine
#                 trap question often has no single-substring answer, so
#                 "review" is a legitimate, expected outcome, not a bug in
#                 the check).

_TCMB_REGIONAL_HOUSING_INDEX_QUESTIONS = [
    {
        "id": "ibbs_region_name_trap",
        "trap": (
            "The obvious-looking answer is wrong: TR10 is Istanbul, not "
            "Balikesir/Canakkale (that's TR22) - a model that pattern-"
            "matches 'TRxx = some Marmara pair of provinces' without "
            "actually reading the file's own 'Seri Aciklamalari' block "
            "(rows 10+, already surfaced in an earlier real answer in this "
            "same session) will get this wrong with total confidence."
        ),
        "ground_truth": "TR22 = Balikesir, Canakkale (verified against the file's own row 17: 'TP.KFE.TR22, TR22 (Balikesir, Canakkale)-Duzey').",
        "question": (
            "Veri setindeki bolgesel kodlarin (TR10, TR22 vb.) Istatistiki "
            "Bolge Birimleri Siniflandirmasi (IBBS) kodlari oldugunu fark "
            "et. TR22 kodunun hangi illeri kapsadigini bul (dosyanin kendi "
            "aciklama bolumunu kullan, tahmin etme). Ardindan TR22 "
            "endeksinin 2021-2026 getirisini 'Balikesir-Canakkale bolgesi' "
            "olarak raporla."
        ),
        "must_include": ["balıkesir", "çanakkale"],
        "must_not_include": ["istanbul"],
    },
    {
        "id": "ibbs_classification_level_mismatch",
        "trap": (
            "TP_KFE_TR7/TR8/TR9/TRA/TRB/TRC are IBBS LEVEL-1 codes (each "
            "covers 6-9 provinces), while TP_KFE_TR10/TR51/TR31/etc are "
            "LEVEL-2 codes (each covers 1-4 provinces) - genuinely different "
            "granularity mixed in the same flat column list. A model that "
            "averages all 21 regional columns as if they were peer, "
            "equal-weight regions is silently comparing a single city "
            "(TR10 = Istanbul alone) against a 9-province aggregate (TRC) "
            "on equal footing."
        ),
        "ground_truth": (
            "Confirmed by hand from the file's own descriptions: TR7 covers "
            "8 provinces (Nevsehir..Yozgat), TRC covers 9 (Gaziantep..Siirt), "
            "while TR10 covers exactly 1 (Istanbul) and TR51 exactly 1 "
            "(Ankara) - NOT the same classification level, despite the "
            "shared 'TRxx' naming pattern."
        ),
        "question": (
            "Sutun isimlerindeki tum bolgesel kodlari (TR10, TR51, TR31, "
            "TR21, TR22, TR32, TR33, TR41, TR42, TR52, TR61, TR62, TR63, "
            "TR7, TR8, TR9, TRA, TRB, TRC) listele. Bunlarin hepsi ayni "
            "Istatistiki Bolge Birimleri Siniflandirmasi (IBBS) seviyesinde "
            "mi, yoksa farkli seviyeler mi karisik? Bu 21 sutunun basit "
            "ortalamasini alarak 'Turkiye genelinde bolgesel KFE ortalamasi' "
            "hesaplamak metodolojik olarak neden yanlis olabilir, acikla."
        ),
        "must_include": [],  # no single substring proves this - review by hand
        "must_not_include": [],
    },
    {
        "id": "observation_method_mismatch",
        "trap": (
            "The file's own metadata (rows 11 and 12) states two DIFFERENT "
            "observation conventions in the same workbook: "
            "TP.ALTINPIYASA.KAP02 is 'Gozlem Degeri: Bitis' (period-END "
            "value) while TP.KFE.TR is 'Gozlem Degeri: Aritmetik Ortalama' "
            "(period-AVERAGE value). An earlier real answer in this exact "
            "session computed a gold-vs-housing CAGR/real-return comparison "
            "WITHOUT ever surfacing this - directly comparing a period-end "
            "price to a period-average index as if they were the same kind "
            "of number."
        ),
        "ground_truth": (
            "Confirmed: row 11's third cell literally reads 'Gozlem Degeri: "
            "Bitis' for gold, row 12's reads 'Gozlem Degeri: Aritmetik "
            "Ortalama' for KFE - both present verbatim in the file."
        ),
        "question": (
            "TP_ALTINPIYASA_KAP02 (altin) ve TP_KFE_TR (konut endeksi) "
            "serilerinin 'Seri Aciklamalari' bolumunde belirtilen gozlem "
            "yontemleri (Gozlem Degeri sutunu) ayni mi, farkli mi? Ikisini "
            "doğrudan karsilastirarak bir getiri/CAGR analizi yaparken bu "
            "fark onemli mi, aciklaya biliyor musun?"
        ),
        "must_include": ["bitiş", "ortalama"],
        "must_not_include": [],
    },
    {
        "id": "nonexistent_region_code",
        "trap": (
            "TR11 does not exist as a column in this file (real level-2 "
            "codes present are TR10/TR21/TR22/TR31/TR32/TR33/TR41/TR42/"
            "TR51/TR52/TR61/TR62/TR63, plus level-1 TR7/TR8/TR9/TRA/TRB/"
            "TRC) - a model must say the column/series does not exist, "
            "not silently substitute a nearby code's data or invent a "
            "number."
        ),
        "ground_truth": "TP_KFE_TR11 is not a column in this dataset - confirmed against the real 22-column header.",
        "question": "TP_KFE_TR11 bolgesinin 2021'den 2026'ya buyume orani nedir?",
        "must_include": [],
        "must_not_include": [],  # checked via admits_missing below instead
        "expect_admits_missing": True,
    },
    {
        "id": "gold_vs_kfe_national_race",
        "trap": (
            "A straightforward analytical (non-trap) CAGR-race question, "
            "kept to verify the numeric-grounding basics still work on "
            "this specific file, not just the trickier questions above."
        ),
        "ground_truth": "Computed from the real 2021 and 2026 values of TP_ALTINPIYASA_KAP02 and TP_KFE_TR in this annual file.",
        "question": "2021'den 2026'ya, altin (TP_ALTINPIYASA_KAP02) mi yoksa Turkiye geneli Konut Fiyat Endeksi (TP_KFE_TR) mi daha hizli buyudu? Yuzde olarak goster.",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "unit_of_gold_series_trap",
        "trap": (
            "The file's own description says TP.ALTINPIYASA.KAP02 is "
            "TL/kg (per kilogram), not TL/gram - a common real-world "
            "confusion since retail gold prices are usually quoted per "
            "gram. A model that assumes gram without checking the file's "
            "own unit would be off by a factor of 1000."
        ),
        "ground_truth": "Row 11 of the file's own 'Seri Aciklamalari' states 'TL/kg-Duzey' - per kilogram, not per gram.",
        "question": "TP_ALTINPIYASA_KAP02 serisinin birimi TL/gram mi, TL/kg mi? Dosyanin kendi aciklamasini kullanarak dogrula.",
        "must_include": ["kg"],
        "must_not_include": [],
    },
]

# --- file 2: same gold+regional-housing base, widened with TCMB's
# household/institutional-sector financial accounts (Finansal Hesaplar) -
# quarterly, 1998-Q4 onward. Real quirks verified by hand: the date
# column reads "1998-4" for the first row (quarter notation, not April);
# all 19 asset/liability sub-items (VF.1-VF.8 assets, F.1-F.8 liabilities)
# are for ONE sector only - "Parasal ve Finansal Kuruluslar" (monetary and
# financial institutions) - in FOUR variants: {Konsolide, Konsolide
# Olmayan} x {Stok, Akim}. There is NO household ("Hanehalki") sector
# series in this file at all, despite that being an easy guess.
_TCMB_REGIONAL_HOUSING_AND_MFI_ACCOUNTS_QUESTIONS = [
    {
        "id": "quarterly_year_notation_trap",
        "trap": (
            "The file's first date value is the literal string '1998-4' - "
            "ambiguous between 'April 1998' (a month) and 'Q4 1998' (a "
            "quarter). The row-count pattern (113 real data rows from "
            "1998 to today) only fits a quarterly series, not monthly - a "
            "model that reads '1998-4' as April and reports 'monthly data "
            "since April 1998' would be wrong on the file's actual "
            "frequency."
        ),
        "ground_truth": "113 real data rows spanning 1998-2026 is consistent with quarterly frequency (~28 years x 4), not monthly (~28 years x 12 would be ~336 rows).",
        "question": (
            "Bu dosyadaki tarih sutunu 'YYYY-N' formatinda degerler "
            "iceriyor (ilk deger '1998-4' gibi). Bu formatin ceyreklik mi "
            "yoksa aylik mi oldugunu, dosyadaki toplam veri satiri sayisina "
            "bakarak nasil belirleyebilecegini gostererek acikla."
        ),
        "must_include": ["çeyrek"],
        "must_not_include": ["aylık veri", "nisan 1998"],
    },
    {
        "id": "consolidated_vs_nonconsolidated_confusion",
        "trap": (
            "Four distinct BF.9 (Net Finansal Deger) series exist for the "
            "SAME sector, differing only by 'Konsolide' vs 'Konsolide "
            "Olmayan' (whether intra-sector claims are netted out) and "
            "'Stok' vs 'Akim'. A model that picks whichever one it finds "
            "first and reports it as 'the' net financial worth, without "
            "noting which of the four it used, is hiding a real "
            "methodological choice from the reader."
        ),
        "ground_truth": (
            "Verified: TP.FINHESTNKS12T.ZP1 (Konsolide Olmayan, Stok), "
            "TP.FINHESTNKS62012T.ZP1 (Konsolide Olmayan, Akim), "
            "TP.FINHESTNKS71012T.ZP1 (Konsolide, Stok), "
            "TP.FINHESTNKS61012T.ZP1 (Konsolide, Akim) are four DIFFERENT "
            "series, all labeled 'BF.9.NET FINANSAL DEGER'."
        ),
        "question": (
            "TP_FINHESTNKS12T_ZP1, TP_FINHESTNKS62012T_ZP1, "
            "TP_FINHESTNKS71012T_ZP1 ve TP_FINHESTNKS61012T_ZP1 "
            "sutunlarinin hepsi 'Net Finansal Deger' adini tasiyor. Bunlar "
            "ayni seri mi, yoksa aralarinda gercek bir metodolojik fark "
            "var mi? Varsa farki acikla."
        ),
        "must_include": ["konsolide"],
        "must_not_include": [],
    },
    {
        "id": "nonexistent_household_sector",
        "trap": (
            "This file's financial-accounts columns are ALL for the "
            "'Parasal ve Finansal Kuruluslar' (monetary/financial "
            "institutions) sector - there is no household ('Hanehalki') "
            "sector series in it at all. A model must say so, not "
            "silently substitute the MFI-sector numbers as if they "
            "answered a household-sector question."
        ),
        "ground_truth": "No column in this file represents the household ('Hanehalki') sector - confirmed against all 19 x 4 = 76 financial-account series descriptions, every one of them 'Parasal ve Finansal Kuruluslar'.",
        "question": "Hanehalkinin (Hanehalki sektörü) net finansal varlik/borç durumunu bu dosyadan bulabilir misin?",
        "must_include": [],
        "must_not_include": [],
        "expect_admits_missing": True,
    },
    {
        "id": "asset_liability_identity_check",
        "trap": (
            "VF.0 (Toplam Varliklar), F.0 (Toplam Yukumlulukler) and BF.9 "
            "(Net Finansal Deger) are not three independent numbers - "
            "national-accounts identity says BF.9 should equal VF.0 minus "
            "F.0 (assets minus liabilities), for the same sector/stok-akim "
            "variant. A model that reports all three without checking "
            "whether they are internally consistent is missing a free "
            "sanity check the data itself offers."
        ),
        "ground_truth": "Standard flow-of-funds identity: Net Financial Worth = Total Assets - Total Liabilities, for the same sector and Stok/Akim variant.",
        "question": (
            "TP_FINHESTNKS12T_ZP23 (Toplam Varliklar, VF.0), "
            "TP_FINHESTNKS12T_ZP45 (Toplam Yukumlulukler, F.0) ve "
            "TP_FINHESTNKS12T_ZP1 (Net Finansal Deger, BF.9) degerlerini "
            "en guncel donem icin sorgula. Varlik eksi yukumlulugun net "
            "finansal degere esit olup olmadigini dogrula - tutarli mi?"
        ),
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "multi_decade_kfe_history_advantage",
        "trap": (
            "This file's TP_KFE_TR history starts 1998, decades earlier "
            "than the annual gold+housing file's 2021 start - a question "
            "asking for a long-run CAGR should use THIS file's deeper "
            "history, not the shorter one, and should recognize the "
            "earlier file simply cannot answer a pre-2021 question."
        ),
        "ground_truth": "This file's TP_KFE_TR column has real observations back to 1998Q4-adjacent quarters, decades before the other uploaded gold+housing file's 2021 start.",
        "question": (
            "Turkiye geneli Konut Fiyat Endeksi'nin (TP_KFE_TR) en eski "
            "veriden bugune kadarki toplam buyume oranini hesapla. Hangi "
            "tabloyu/dosyayi kullandigini ve neden onu sectigini belirt."
        ),
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "mfi_stok_akim_magnitude_sanity",
        "trap": (
            "Stok (a level, e.g. total assets outstanding) and Akim (a "
            "flow, e.g. the quarterly change) are different kinds of "
            "quantity - a model that compares their raw magnitudes "
            "directly ('the flow figure is X% of the stock figure, so...') "
            "without noting one is a level and the other a period change "
            "is comparing apples to oranges."
        ),
        "ground_truth": "TP_FINHESTNKS12T_ZP1 is a Stok (level, outstanding at period-end); TP_FINHESTNKS62012T_ZP1 is an Akim (flow, the period's change) - not directly comparable magnitudes.",
        "question": "Net Finansal Deger'in Stok (TP_FINHESTNKS12T_ZP1) ve Akim (TP_FINHESTNKS62012T_ZP1) degerlerini karsilastir. Akim, stogun ne kadari? Bu oranlama anlamli mi?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "mfi_liability_breakdown_largest_item",
        "trap": (
            "A straightforward analytical question (not a trap) to keep "
            "this set honest about non-adversarial cases too: among the "
            "eight F.1-F.8 liability sub-items, which is actually the "
            "largest at the latest date - a real number-comparison task, "
            "not a methodology gotcha."
        ),
        "ground_truth": "Determined by actually querying F.1 (ZP24) through F.8 (ZP44) at the latest real date and comparing.",
        "question": "Parasal ve Finansal Kuruluslarin (Konsolide Olmayan, Stok) yukumluluk kalemleri arasinda (F.1'den F.8'e) en guncel tarihte en buyuk kalem hangisi?",
        "must_include": [],
        "must_not_include": [],
    },
]

# --- file 3: Balance of Payments (Odemeler Dengesi), "Arsiv" tag on every
# series description - TCMB's legacy BPM5-methodology archive series,
# discontinued after Turkey's move to BPM6. Real data verified by hand to
# stop at 2015-02 - it cannot answer anything about a later period, and
# presenting its numbers as "the balance of payments" without the
# methodology caveat would be misleading even for the years it covers.
_TCMB_BALANCE_OF_PAYMENTS_ARCHIVE_QUESTIONS = [
    {
        "id": "archive_methodology_and_date_range_trap",
        "trap": (
            "Real data verified by hand to stop at 2015-02 - a question "
            "about 2018-2024 trends cannot be answered from this file at "
            "all, AND every series here is explicitly tagged '(Arsiv)' in "
            "its own description (TCMB's legacy BPM5 methodology, "
            "superseded by BPM6) - a model should ideally flag BOTH the "
            "missing years and that even the covered years are a "
            "discontinued methodology, not silently answer with old "
            "numbers as if current."
        ),
        "ground_truth": "Verified: last real data row is 2015-02; every series description ends in '(Arsiv)-Duzey'.",
        "question": "2018'den 2024'e kadar Turkiye'nin cari islemler dengesi (TP_OD_Z01) nasil bir trend izledi?",
        "must_include": [],
        "must_not_include": [],
        "expect_admits_missing": True,
    },
    {
        "id": "current_account_identity_check",
        "trap": (
            "A-b (Mal ve Hizmet Dengesi, Z07) is not an independent "
            "number - it is defined as A-a (Mal Dengesi, Z04) plus the "
            "services balance (Hizmet Geliri Z05 minus Hizmet Gideri Z06). "
            "A model that reports Z07 without ever checking it against "
            "Z04/Z05/Z06 misses a free internal-consistency check."
        ),
        "ground_truth": "Balance-of-payments identity: A-b (Mal ve Hizmet Dengesi) = A-a (Mal Dengesi) + (Hizmet Geliri - Hizmet Gideri).",
        "question": (
            "TP_OD_Z04 (Mal Dengesi), TP_OD_Z05 (Hizmetler: Gelir), "
            "TP_OD_Z06 (Hizmetler: Gider) ve TP_OD_Z07 (Mal ve Hizmet "
            "Dengesi) degerlerini ayni donem icin sorgula. Z07'nin "
            "digerlerinden turetilip turetilemedigini kontrol et."
        ),
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "net_errors_omissions_meaning",
        "trap": (
            "'Net Hata ve Noksan' (Z31) is a statistical-discrepancy "
            "plug, not a real economic flow - a model must not analyze "
            "its trend as if it were a meaningful indicator of anything "
            "(e.g. 'capital flight'), only as a measurement-quality "
            "signal (a large or growing value questions the reliability "
            "of the other components, it doesn't describe the economy)."
        ),
        "ground_truth": "TP.OD.Z31 = 'D-NET HATA VE NOKSAN' - the standard BoP statistical discrepancy line, not an economic aggregate.",
        "question": "TP_OD_Z31 (Net Hata ve Noksan) serisi ne anlama gelir? Bu serinin buyumesi Turkiye ekonomisi hakkinda dogrudan bir sey soyler mi?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "reserve_assets_vs_imf_credit_trap",
        "trap": (
            "TP_OD_Z34 (Rezerv Varliklar) and TP_OD_Z35 (Uluslararasi "
            "Para Fonu Kredileri) are different E-section sub-items (E-14 "
            "vs E-15) - a model asked about 'reserves' must not silently "
            "fold in IMF credit flows as if they were the same thing."
        ),
        "ground_truth": "TP.OD.Z34 = 'E-14-Rezerv Varliklar', TP.OD.Z35 = 'E-15-Uluslararasi Para Fonu Kredileri' - two distinct sub-items under section E.",
        "question": "TP_OD_Z34 (Rezerv Varliklar) ile TP_OD_Z35 (IMF Kredileri) ayni kalemin parcasi mi, yoksa ayri mi? Karistirmadan ikisini de aciklar misin?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "capital_vs_financial_account_trap",
        "trap": (
            "'Sermaye Hesabi' (Capital Account, Z12/B) and 'Finans "
            "Hesabi' (Financial Account, Z13/C) are two DIFFERENT BPM "
            "top-level components (the Capital Account is small - mostly "
            "debt forgiveness/non-produced assets - while the Financial "
            "Account covers direct/portfolio/other investment) - a model "
            "must not treat 'sermaye hareketleri' as a synonym for either "
            "one specifically without checking which the question means."
        ),
        "ground_truth": "TP.OD.Z12 = 'B-SERMAYE HESABI', TP.OD.Z13 = 'C-FINANS HESABI' - distinct BPM top-level accounts, not synonyms.",
        "question": "TP_OD_Z12 (Sermaye Hesabi) ile TP_OD_Z13 (Finans Hesabi) arasindaki fark nedir? Ikisi ayni sey mi?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "portfolio_liability_breakdown_sanity",
        "trap": (
            "A straightforward analytical (non-trap) question: split "
            "Portfoy Hesabi-Yukumlulukler (Z17) into its two named "
            "sub-items, Hisse Senetleri (Z18) and Borc Senetleri (Z19), "
            "and check whether they actually sum close to the parent "
            "total - a real arithmetic check against real numbers."
        ),
        "ground_truth": "TP.OD.Z18 (Hisse Senetleri) + TP.OD.Z19 (Borc Senetleri) should sum close to TP.OD.Z17 (Portfoy Hesabi-Yukumlulukler), the parent category.",
        "question": "TP_OD_Z18 (Hisse Senetleri) ile TP_OD_Z19 (Borc Senetleri) toplami, TP_OD_Z17 (Portfoy Hesabi-Yukumlulukler) degerine yakin mi? Gercek sayilarla dogrula.",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "sector_breakdown_of_other_investment",
        "trap": (
            "Diger Yatirimlar-Yukumlulukler (Z25) breaks down into FOUR "
            "sectors: Merkez Bankasi (Z26), Genel Hukumet (Z27), Bankalar "
            "(Z28), Diger Sektorler (Z29) - a model asked which sector "
            "carries the most external liability must actually compare "
            "all four, not assume 'Merkez Bankasi' by default."
        ),
        "ground_truth": "TP.OD.Z26/Z27/Z28/Z29 are the sector breakdown of TP.OD.Z25 (Diger Yatirimlar-Yukumlulukler) - Merkez Bankasi, Genel Hukumet, Bankalar, Diger Sektorler respectively.",
        "question": "Diger Yatirimlar-Yukumlulukler icinde (Merkez Bankasi, Genel Hukumet, Bankalar, Diger Sektorler) hangi sektor en buyuk paya sahip? Dorduncu sectoru de unutma.",
        "must_include": [],
        "must_not_include": [],
    },
]

# --- file 4: consumer/commercial loan and TRY/EUR/USD deposit interest
# rates, WEEKLY frequency, dates as 'DD-MM-YYYY' strings (e.g.
# '09-07-2021') - genuinely ambiguous against a US-style MM-DD-YYYY
# reading, and directly relevant to the KKB/credit theme this whole
# project is built around.
_TCMB_LOAN_AND_DEPOSIT_RATES_QUESTIONS = [
    {
        "id": "ambiguous_weekly_date_format",
        "trap": (
            "'09-07-2021' is genuinely ambiguous: 9 July 2021 (DD-MM, the "
            "EVDS/Turkish convention) or 7 September 2021 (MM-DD, the US "
            "convention) both parse. A model must state which convention "
            "it used and why (weekly EVDS exports are always DD-MM-YYYY), "
            "not silently pick one."
        ),
        "ground_truth": "EVDS exports are DD-MM-YYYY by convention - the file's first row '09-07-2021' is 9 July 2021, not 7 September.",
        "question": "Bu dosyadaki tarihler '09-07-2021' gibi yaziliyor. Bu 9 Temmuz mu, yoksa 7 Eylul mu? Hangi tarih formatini varsaydigini acikca belirt.",
        "must_include": ["temmuz"],
        "must_not_include": ["eylül"],
    },
    {
        "id": "loan_deposit_spread_kkb_relevant",
        "trap": (
            "A meaningful credit-market question for a KKB-themed system: "
            "the spread between consumer loan rates and deposit rates is "
            "a real risk/margin indicator - a model should compute it as "
            "an actual subtraction of two real series, not describe it "
            "qualitatively without numbers."
        ),
        "ground_truth": "TP_KTFTUK (tuketici kredisi, ihtiyac+tasit+konut birlesik) and TP_TRY_MT06 (TL toplam mevduat faizi) are both present and directly comparable.",
        "question": "Tuketici kredisi faizi (TP_KTFTUK) ile TL toplam mevduat faizi (TP_TRY_MT06) arasindaki makas en guncel veride kac puan? Zaman icinde acildi mi kapandi mi?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "currency_carry_reasoning",
        "trap": (
            "TRY deposit rates run far above EUR/USD deposit rates in "
            "this data - a model should connect this to Turkish inflation "
            "expectations / uncovered interest parity, not just report "
            "the gap as an unexplained number."
        ),
        "ground_truth": "TL toplam mevduat faizi (TP_TRY_MT06) trades far above EUR (TP_EUR_MT06) and USD (TP_USD_MT06) equivalents at every date in this file.",
        "question": "TL mevduat faizi neden EUR ve USD mevduat faizinden bu kadar yuksek? Sayilarla goster ve ekonomik nedenini acikla.",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "deposit_maturity_curve_shape",
        "trap": (
            "TP_TRY_MT01 through MT05 are 1-month through 1-year+ "
            "maturities of the SAME currency/product - a model asked "
            "about the 'yield curve shape' must compare across maturities "
            "at ONE date, not conflate a maturity comparison with a "
            "time-series trend."
        ),
        "ground_truth": "TP_TRY_MT01 (1 ay), MT02 (3 ay), MT03 (6 ay), MT04 (1 yil), MT05 (1 yil+) are maturity bands of the same TL deposit rate at a single point in time.",
        "question": "En guncel tarihte TL mevduat faiz egrisi (1 ay, 3 ay, 6 ay, 1 yil, 1 yil+) yukselen mi alcalan mi? Bu ne anlama gelir?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "currency_suffix_averaging_trap",
        "trap": (
            "TP_KTF17, TP_KTF17_EUR and TP_KTF17_USD are the SAME "
            "'Ticari Krediler' product in three DIFFERENT currencies - "
            "averaging or summing them together would be meaningless "
            "(mixing a TL rate near 50%+ with a USD rate near 7-8% "
            "produces a number with no real interpretation)."
        ),
        "ground_truth": "TP.KTF17 (TL), TP.KTF17.EUR, TP.KTF17.USD are the same 'Ticari Krediler' series broken out by currency - not three independent products to be averaged.",
        "question": "TP_KTF17, TP_KTF17_EUR ve TP_KTF17_USD serilerinin ortalamasini alip 'ticari kredi faizi ortalama %X' diye raporlayabilir miyim? Neden olur ya da olmaz?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "kmh_included_vs_excluded_trap",
        "trap": (
            "TP_KTF10 (ihtiyac kredisi) and TP_KTF101 (ihtiyac kredisi, "
            "KMH DAHIL - overdraft included) are near-identical-sounding "
            "but structurally different products; same for "
            "TP_KTFTUK/TP_KTFTUK01. A model must not treat the KMH-dahil "
            "and KMH-haric variants as duplicates or average them."
        ),
        "ground_truth": "TP.KTF10 excludes KMH (overdraft), TP.KTF101 includes it - two distinct rate series, not duplicates.",
        "question": "TP_KTF10 ile TP_KTF101 arasindaki fark nedir? Bu iki seriyi ayni kredi urununun tekrari sayip birini gormezden gelebilir miyim?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "consumer_loan_rate_five_year_change",
        "trap": (
            "A straightforward, non-trap analytical question (weekly "
            "data, 2021-07 to 2026-09): compute the real change in "
            "TP_KTFTUK over the file's actual full span, using the "
            "correct first/last real values, not assumed ones."
        ),
        "ground_truth": "Determined by actually querying the first and last real TP_KTFTUK values in this weekly series (2021-07-09 through 2026-09-04).",
        "question": "TP_KTFTUK (tuketici kredisi faizi) bu dosyadaki en eski tarihten en guncel tarihe kadar kac puan degisti?",
        "must_include": [],
        "must_not_include": [],
    },
]

_TCMB_TRAVEL_REVENUE_PDF_QUESTIONS = [
    {
        "id": "image_only_pdf_graceful_failure",
        "trap": (
            "This PDF is a single embedded IMAGE (a chart screenshot) "
            "with zero extractable text or table structure (verified: "
            "page.chars, page.lines, page.rects all empty; page.images "
            "has exactly one entry) - pdf_import.py has no OCR fallback. "
            "A model asked for exact figures from it MUST say it cannot "
            "extract data from this file, not invent plausible-looking "
            "travel-revenue numbers."
        ),
        "ground_truth": "Confirmed live: pdfplumber finds 0 chars, 0 lines, 0 rects, 1 image on the single page - this file has no machine-readable table or text at all.",
        "question": "Odemeler Dengesi Seyahat Gelirleri PDF dosyasindan 2023 yilinin aylik seyahat gelirlerini tablo halinde cikar.",
        "must_include": [],
        "must_not_include": [],
        "expect_admits_missing": True,
    },
    {
        "id": "moving_total_misinterpretation",
        "trap": (
            "The file's own title says 'Hareketli Toplam' (moving/rolling "
            "total, a 12-month trailing sum) - even if the model somehow "
            "had the numbers, describing a single point as 'that month's "
            "revenue' or reading a small month-to-month change as a real "
            "swing would misread the metric's construction (it is smooth "
            "by definition, since it's a rolling sum)."
        ),
        "ground_truth": "'Hareketli Toplam' (moving/rolling total) is a smoothed 12-month trailing sum by construction - a single data point is not one month's revenue.",
        "question": "'Odemeler Dengesi Seyahat Gelirleri (Milyon ABD Dolari Hareketli Toplam)' basligindaki 'Hareketli Toplam' ifadesi ne anlama gelir? Bu seriden tek bir ayin gelirini dogrudan okuyabilir miyiz?",
        "must_include": [],
        "must_not_include": [],
    },
]

# --- cross-file questions: require combining the core gold layer
# (gold_fact_long) with one or more of the private uploaded tables above -
# the closest thing in this benchmark to the mentor feedback's own worked
# example ("faiz yukselmesiyle beraber kredi talebinde dusus yasandi -
# dunya piyasalarinin bu duruma etkisi ne oldu" - an internal anomaly
# that should be explainable, and answerable when asked to connect it to
# external/world-market context via web_search).
CROSS_FILE_QUESTIONS = [
    {
        "id": "anomaly_plus_world_market_context",
        "trap": (
            "Directly modeled on real mentor feedback for this project: "
            "an internal finding (loan demand falling as rates rise) "
            "should be explainable AND the model should be able to "
            "connect it to external context (global markets) when asked - "
            "this requires web_search, which was found DOWN in this same "
            "session (SearXNG not running). The honest, correct behavior "
            "when search is unavailable is to say so plainly and answer "
            "only from internal data - NOT to fabricate a 'dunya "
            "piyasalari' narrative with no real source."
        ),
        "ground_truth": "web_search depends on a local SearXNG container that was confirmed down earlier this session - any external-context claim made without it succeeding is unsourced.",
        "question": (
            "Tuketici kredisi faizinde (TP_KTFTUK) son bir yilda yukselis "
            "oldugu donemde kredi hacminde/talebinde bir yavaslama var mi? "
            "Varsa, bu donemde dunya piyasalarinda (FED faiz kararlari, "
            "kuresel risk istahi vb.) neler oldugunu arastirip bu "
            "durumla iliskilendirebilir misin?"
        ),
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "multi_source_join_reasoning",
        "trap": (
            "gold_fact_long already has a housing loan rate series "
            "(TP.BKR.TRY.18) - a model asked to relate it to the "
            "uploaded regional housing-index data must actually JOIN "
            "across the public gold layer and a private uploaded table, "
            "not silently substitute one for the other or refuse because "
            "they live in different tables."
        ),
        "ground_truth": "TP.BKR.TRY.18 (Konut Kredisi Faizi) lives in gold_fact_long; TP_KFE_TR* regional series live only in the uploaded tables - a real answer requires querying both.",
        "question": "Konut kredisi faizi (gold_fact_long, TP.BKR.TRY.18) ile bolgesel Konut Fiyat Endeksi (yuklenen tablo, TP_KFE_TR51 - Ankara) arasinda 2021-2026 arasi bir iliski var mi? Ikisini de gercekten sorgulayarak cevapla.",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "auto_discovery_of_uploaded_tables",
        "trap": (
            "Mirrors the mentor feedback's core ask almost verbatim: given "
            "several already-uploaded TCMB files, can the agent discover "
            "which lakehouse tables exist and combine them on its own, "
            "with NO manual guidance about table names, rather than "
            "needing to be told exactly which table holds which series?"
        ),
        "ground_truth": "list_imported_datasets / lakehouse_schema should surface every uploaded table's real name - a correct answer names the actual tables it used, not a guessed or generic one.",
        "question": "Yuklu olan TCMB dosyalarinin hepsini listele (tablo adlariyla) ve hangi ekonomik konulari kapsadiklarini kisaca ozetle. Manuel olarak hangi tabloyu kullanacagini soylemedim - kendin kesfet.",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "frequency_mismatch_join_trap",
        "trap": (
            "Joining the weekly loan/deposit-rate table against the "
            "monthly gold_fact_long series on raw date equality would "
            "silently drop almost every row (a weekly Wednesday date "
            "rarely equals a monthly YYYY-MM period string) - a model "
            "must recognize the frequency mismatch and aggregate/align "
            "appropriately (e.g. resample to monthly), not report a "
            "near-empty join result as if it were a real answer."
        ),
        "ground_truth": "The loan/deposit-rate table is weekly (DD-MM-YYYY dates); gold_fact_long's Period column is monthly ('YYYY-MM') - a naive equality join between them matches almost nothing.",
        "question": "Haftalik tuketici kredisi faizi (TP_KTFTUK) ile aylik TCMB politika faizini (gold_fact_long, TP.BISPOLFAIZ.TUR) ayni tabloda karsilastir. Iki serinin farkli frekansta oldugunu fark ettin mi, nasil hizaladin?",
        "must_include": [],
        "must_not_include": [],
    },
    {
        "id": "private_vs_public_data_disclaimer",
        "trap": (
            "Every uploaded table must be flagged as user-supplied and "
            "unverified when cited, same discipline as gold_* series get "
            "the opposite ('resmi kaynak') framing - a model that quotes "
            "numbers from an uploaded TCMB file without this caveat is "
            "overstating how trustworthy the number is."
        ),
        "ground_truth": "This project's own design discipline (see src/agent/data_import.py's module docstring) requires every answer citing a user-uploaded table to say plainly that it is unverified, unlike a gold_* series.",
        "question": "Yuklenen TCMB dosyalarindan aldigin sayilar ile gold_fact_long'daki resmi TCMB serilerinin guvenilirlik acisindan bir farki var mi? Cevabinda bu farki nasil belirtirsin?",
        "must_include": [],
        "must_not_include": [],
    },
]


def _column_signature(columns: list[str]) -> str:
    """A stable, content-derived key for a file's schema - sorted,
    lowercased column names, hashed. Two files with the same real columns
    get the same signature regardless of filename; the same file sent
    again under a different name still matches."""
    import hashlib

    normalized = ",".join(sorted(str(c).strip().lower() for c in columns))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# Signatures computed once, by hand, from each real file's own header -
# recompute with _column_signature(list(df.columns)) against any new
# file to add its own entry here. Never a filename.
QUESTION_SETS = {
    "3992b775aab80df5": _TCMB_REGIONAL_HOUSING_INDEX_QUESTIONS,          # Tarih + TP_ALTINPIYASA_KAP02 + TP_KFE_TR(xx) x19, 22 cols, annual
    "34c09d90b6fe7e51": _TCMB_REGIONAL_HOUSING_AND_MFI_ACCOUNTS_QUESTIONS,  # same base + Finansal Hesaplar (MFI sector), 107 cols, quarterly
    "9040baa7b0aab2d3": _TCMB_LOAN_AND_DEPOSIT_RATES_QUESTIONS,          # loan/deposit interest rates, 93 cols, weekly
    "c3e1fc952b19886f": _TCMB_BALANCE_OF_PAYMENTS_ARCHIVE_QUESTIONS,     # Odemeler Dengesi (Arsiv/BPM5), 225 cols, monthly 2002-2015
}
# PDF question sets are NOT keyed by a hash of the file (that would only
# ever match this one exact file again) - they are keyed by a real,
# content-derived STRUCTURAL property instead: whether pdfplumber finds
# any extractable table at all. A PDF with zero tables (a scanned image,
# a chart export) gets _TCMB_TRAVEL_REVENUE_PDF_QUESTIONS regardless of
# which specific such file it is, since the trap they test (graceful
# failure, no fabricated numbers) applies to any image-only PDF equally.
PDF_QUESTIONS_FOR_ZERO_TABLES = _TCMB_TRAVEL_REVENUE_PDF_QUESTIONS

_MISSING_DATA_RE = re.compile(
    r"bulunam|mevcut de[gğ]il|yok\b|eri[sş]ilem|veri(?:si)? yok|"
    r"s[uü]tun(?:u)? yok|kolon(?:u)? yok|tanımlı de[gğ]il",
    re.IGNORECASE,
)


def _import_file(path: Path) -> str:
    """Returns the dataset_name it was imported under."""
    stem = re.sub(r"[^a-zA-Z0-9_]", "_", path.stem).lower()
    dataset_name = f"advtest_{stem}"
    if path.suffix.lower() == ".pdf":
        result = import_pdf_table(path, dataset_name)
    else:
        result = import_dataset(path, dataset_name)
    logger.info(f"_import_file: {path.name} -> '{dataset_name}' ({result['row_count']} rows, {result['column_count']} cols)")
    return dataset_name


def _check_answer(item: dict, answer: str) -> dict:
    lowered = answer.lower()
    must_include = item.get("must_include", [])
    must_not_include = item.get("must_not_include", [])
    include_ok = all(s in lowered for s in must_include)
    exclude_ok = all(s not in lowered for s in must_not_include)

    if item.get("expect_admits_missing"):
        admitted = bool(_MISSING_DATA_RE.search(lowered))
        return {"verdict": "PASS" if admitted else "FAIL", "detail": f"admits_missing={admitted}"}

    if not must_include and not must_not_include:
        return {"verdict": "REVIEW", "detail": "No auto-checkable substring for this trap - read the answer by hand."}

    verdict = "PASS" if (include_ok and exclude_ok) else "FAIL"
    return {
        "verdict": verdict,
        "detail": f"must_include_ok={include_ok}, must_not_include_ok={exclude_ok}",
    }


def _run_questions(questions: list[dict]) -> list[dict]:
    results = []
    for item in questions:
        logger.info(f"_run_questions: asking '{item['id']}'")
        t0 = time.monotonic()
        answer, trace = ask(item["question"], return_trace=True)
        elapsed = round(time.monotonic() - t0, 1)
        check = _check_answer(item, answer)
        results.append({
            "id": item["id"], "trap": item["trap"], "ground_truth": item["ground_truth"],
            "question": item["question"], "answer": answer, "elapsed_sec": elapsed,
            "tools_called": [t["name"] for t in trace], "auto_check": check,
        })
        logger.info(f"  [{check['verdict']}] {item['id']} ({elapsed}s)")
    return results


def run_file(path: Path) -> dict:
    logger.info(f"run_file: {path.name}")

    if path.suffix.lower() == ".pdf":
        preview = list_pdf_tables(path)
        if preview["tables"]:
            schema = None
            dataset_name = _import_file(path)
            questions = []  # no hand-crafted set for a PDF that DOES have real tables yet
            row_count = col_count = None
            data_quality = {}
        else:
            logger.warning(f"run_file: {path.name} has 0 extractable tables (image-only or unstructured PDF) - running the zero-table question set instead of importing.")
            dataset_name = None
            questions = PDF_QUESTIONS_FOR_ZERO_TABLES
            row_count = col_count = 0
            data_quality = {"note": "0 extractable tables - see list_pdf_tables preview"}
        results = _run_questions(questions)
        return {
            "file": path.name, "dataset_name": dataset_name,
            "schema": {"row_count": row_count, "column_count": col_count, "data_quality": data_quality},
            "results": results,
        }

    schema = detect_schema(path)
    dataset_name = _import_file(path)

    signature = _column_signature([c["column"] for c in schema["columns"]])
    questions = QUESTION_SETS.get(signature, [])
    if not questions:
        logger.warning(
            f"run_file: no hand-crafted question set for column-signature {signature!r} "
            f"(file: {path.name}) - imported and schema-checked only. Add an entry to "
            f"QUESTION_SETS keyed by this signature to give it its own trap questions."
        )

    results = _run_questions(questions)

    return {
        "file": path.name, "dataset_name": dataset_name,
        "schema": {"row_count": schema["row_count"], "column_count": schema["column_count"],
                   "data_quality": schema["data_quality"]},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial benchmark against private real Excel/CSV/PDF files in test_excels/")
    parser.add_argument("--file", help="Only run the file in test_excels/ with this stem (filename without extension)")
    parser.add_argument("--skip-cross-file", action="store_true", help="Skip the cross-file question set (needs all files already imported)")
    args = parser.parse_args()

    if not TEST_EXCELS_DIR.exists():
        logger.error(f"main: {TEST_EXCELS_DIR} does not exist - nothing to test.")
        raise SystemExit(1)

    files = sorted(
        p for p in TEST_EXCELS_DIR.iterdir()
        if p.suffix.lower() in (".xlsx", ".xls", ".csv", ".pdf") and (not args.file or p.stem == args.file)
    )
    if not files:
        logger.error(f"main: no supported file found in {TEST_EXCELS_DIR} (looked for .xlsx/.xls/.csv/.pdf).")
        raise SystemExit(1)

    reports = [run_file(p) for p in files]

    cross_file_results = []
    if not args.skip_cross_file and not args.file:
        logger.info(f"main: running {len(CROSS_FILE_QUESTIONS)} cross-file question(s)")
        cross_file_results = _run_questions(CROSS_FILE_QUESTIONS)

    total_questions = sum(len(r["results"]) for r in reports) + len(cross_file_results)
    total_pass = sum(1 for r in reports for item in r["results"] if item["auto_check"]["verdict"] == "PASS")
    total_pass += sum(1 for item in cross_file_results if item["auto_check"]["verdict"] == "PASS")
    total_fail = sum(1 for r in reports for item in r["results"] if item["auto_check"]["verdict"] == "FAIL")
    total_fail += sum(1 for item in cross_file_results if item["auto_check"]["verdict"] == "FAIL")
    total_review = total_questions - total_pass - total_fail

    output = {
        "summary": {"total_questions": total_questions, "pass": total_pass, "fail": total_fail, "review": total_review},
        "files": reports,
        "cross_file": cross_file_results,
    }
    REPORT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("ADVERSARIAL EXCEL/PDF BENCHMARK (private test_excels/ files)")
    print("=" * 70)
    for r in reports:
        print(f"\n{r['file']}  ({r['schema']['row_count']} rows, {r['schema']['column_count']} cols)")
        if not r["results"]:
            print("  (no hand-crafted question set for this file)")
        for item in r["results"]:
            print(f"  [{item['auto_check']['verdict']:6s}] {item['id']:35s} {item['elapsed_sec']:5.1f}s")
    if cross_file_results:
        print("\nCROSS-FILE:")
        for item in cross_file_results:
            print(f"  [{item['auto_check']['verdict']:6s}] {item['id']:35s} {item['elapsed_sec']:5.1f}s")
    print("-" * 70)
    print(f"TOPLAM: {total_questions} soru | PASS={total_pass} FAIL={total_fail} REVIEW={total_review}")
    print(f"Full report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
