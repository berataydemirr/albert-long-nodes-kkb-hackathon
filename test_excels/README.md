# test_excels/ — real TCMB EVDS exports for adversarial benchmarking

These are real, publicly available exports from TCMB's EVDS (Elektronik
Veri Dağıtım Sistemi) portal — not synthetic fixtures, not proprietary or
personal data. They exist so `benchmarks/adversarial_excel_benchmark.py`
can be tested against genuinely messy real-world files (mixed date
formats, merged headers, legacy/discontinued methodologies, unusual
units) instead of clean hand-built test data.

On `main-temp-with-feedback` and other branches, this folder is
gitignored (private-by-default is the safer starting point for any
user-uploaded file). This branch (`main-temp-excel-benchmarks`) is the
one exception, added deliberately because these specific 5 files are
confirmed public TCMB statistical data.

## Files

| File | Content | Real quirk it tests |
|---|---|---|
| `tcmb_altin_ve_bolgesel_konut_fiyat_endeksi_yillik.xlsx` | Gold price (TL/kg) + national/regional Housing Price Index, annual, 2021-2026 | İBBS region-code↔name mapping, differing observation conventions (period-end vs period-average) |
| `tcmb_bolgesel_konut_endeksi_ve_parasal_finansal_kuruluslar_hesaplari.xlsx` | Same base, widened with Monetary/Financial Institutions flow-of-funds accounts, quarterly, 1998Q4+ | Ambiguous quarter notation, Consolidated vs Non-consolidated accounting |
| `tcmb_kredi_ve_mevduat_faiz_oranlari.xlsx` | Consumer/commercial loan and TRY/EUR/USD deposit interest rates, weekly, 2021-2026 | DD-MM-YYYY date ambiguity, currency-suffix averaging trap |
| `tcmb_odemeler_dengesi_arsiv_bpm5.xlsx` | Full Balance of Payments statement, monthly, 2002-2015 | Legacy BPM5 "Arşiv" methodology (discontinued after BPM6), accounting identities |
| `tcmb_odemeler_dengesi_seyahat_gelirleri_hareketli_toplam.pdf` | Balance of Payments travel revenue chart | Image-only PDF with zero extractable table structure — tests graceful failure, not fabrication |

`_superseded/` holds earlier, narrower/duplicate re-exports of the same
underlying series (kept locally for reference, not committed) — the 5
files above are each the widest/most complete version of their lineage.

## Running the benchmark

```
python -m benchmarks.adversarial_excel_benchmark
python -m benchmarks.adversarial_excel_benchmark --file <stem>
```

Drop your own additional real files in here (any format the project's
import pipeline supports: `.xlsx`, `.xls`, `.csv`, `.pdf`) to exercise
the pipeline against them too — a file with no hand-crafted question set
in `adversarial_excel_benchmark.py` still gets imported and schema-checked,
just without the trap questions. Question sets are keyed by a hash of the
file's own column names (or, for PDFs with no extractable table, applied
by that structural property) — never by filename — so add your own entry
to `QUESTION_SETS` keyed the same way if you want hand-crafted traps for
a new file.
