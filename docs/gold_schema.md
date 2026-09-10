# Gold Layer — Schema Notes

Running `python -m src.gold.build_gold` writes 5 tables into
`data/gold/lakehouse.duckdb`. The Lakehouse Tool queries these tables.

## ⚠️ Critical: always pass `encoding="utf-8"` for CSV read/write

On this machine (Windows, Turkish locale) the **default** `pd.read_csv(path)` /
`df.to_csv(path)` calls use the OS's local code page (cp1254). The CSVs on
disk are genuinely correct UTF-8 (verified against the bronze JSON bytes),
but when read without `encoding="utf-8"` the Turkish characters in words
like "DÖNEM" silently turn into `�`. This is the kind of bug that goes
unnoticed on the author's own machine and shows up as garbled text in front
of the jury during the demo.
**Every new piece of code that touches this data (tools included) must use
`pd.read_csv(path, encoding="utf-8")` and `df.to_csv(path, encoding="utf-8")`.**
The existing `src/processing/*.py` and `src/gold/*.py` scripts were fixed for this.

Note: you can still see `�` in terminal/log output (e.g. `İSTANBUL` shows
as `�STANBUL` in the console) - this is ONLY the console's code page
issue; checking with `repr()` / the hex codepoint confirms the data itself
is correct (`0x130` = `İ`). Don't assume the data is corrupt from console
output alone; the real check is the bytes on disk/in the DB.

## Tables

### `gold_fact_long`
Long/tidy format across all sources (EVDS, TUIK, BDDK Monthly, BDDK
Weekly, BDDK FinTurk-national) on a shared monthly axis: one row = one
observation. General-purpose - the table a natural-language→SQL tool
should actually be pointed at.

| Column | Description |
|---|---|
| `Period` | `YYYY-MM` |
| `Bolge` | `TR` (country-wide) - see `gold_finturk_regional_quarterly` for province-level data |
| `Kaynak` | `evds` / `tuik` / `bddk_monthly` / `bddk_weekly` / `bddk_finturk` |
| `Seri_Kod` | Source-specific series/table code (e.g. `TP.KFE.TR`, `tablo_4.konut_kredisi`) |
| `Seri_Adi` | Human-readable description (this is what gets shown to the LLM in the prompt) |
| `Deger` | Numeric value |
| `Birim` | `%`, `milyon TL` (million TRY), `TRY`, `Bin TL (dogrulanmadi)` (thousand TRY, unconfirmed), empty/`NULL` = an index or the unit couldn't be confirmed |
| `Tip` | `stock` (level at period end) / `flow` (flow for that month) / `flow_quarter_ffill` (forward-filled from quarter-end to month, not a real monthly observation) |
| `Kaynak_Dosya` | Traceability: which silver file this row came from |

**24 series, 1584 rows** (2021-01 → 2026-06), 24/24 series with full
coverage (66/66 periods, zero nulls - checked on every build against
`gold_fact_long`, and `tests/test_gold_data_integrity.py` also runs
against the real build). By source:

- **EVDS (11 series):** house price index, housing loan interest rate,
  TCMB policy rate, CPI, industrial production index, USD/TRY,
  TRY deposit interest rate, credit/debit card spending (averaged
  from weekly to monthly), real-sector confidence index, M2 money
  supply (weekly to monthly), combined consumer loan interest rate
  (needs+auto+housing combined). All found via `--discover` against
  real data, none guessed (see the comments in `config/evds_series.py`).
- **TUIK (1 series):** unemployment rate.
- **BDDK Monthly (10 series):** housing credit (tablo_4), **total
  loans** (tablo_3), **total deposits** (tablo_9), **capital adequacy
  standard ratio** (tablo_12, `%`), **total assets**, **total
  liabilities**, **total equity** (tablo_1/Balance Sheet), **total SME
  loans** (tablo_6), **liquidity coverage ratio** (tablo_11, `%`),
  **non-performing loan ratio / NPL** (tablo_15, `%`). Each was matched
  exactly against the real `Column_2` text and confirmed "not
  cumulative" against `cumulative_scan_report_bddk_monthly.csv` - none
  of these needed a new network call, bronze/silver were already
  complete on disk (1122/1122 files). Total assets/liabilities were
  cross-checked against the balance sheet's own accounting identity:
  assets == liabilities in every one of the 66 months (diff = 0), see
  `tests/test_gold_data_integrity.py::test_bddk_balance_sheet_assets_equal_liabilities`.
  **Deliberately left out:** tablo_15's ROA/ROE-style rows ("period net
  profit / average total assets (%)" etc.) came back `is_cumulative=True`
  in the scan report - since their numerator (period net profit) is
  itself YTD-cumulative, the ratio behaves as YTD-cumulative too, and a
  proper month-by-month figure would require decumulating the numerator
  first - not added (reporting it wrong was worse than not having it).
  **tablo_6 (SME Loans) caused a separate bug:** this table has no
  column named exactly "Toplam" (it has sub-columns instead: domestic/
  foreign-currency cash loans, past-due loans, customer count, etc.),
  and `load_table_long()`'s generic "last numeric column" fallback
  silently picked the customer COUNT (0 in most months) - caught live
  (it was reporting 0.0 TL for all 66 months), fixed with a dedicated
  loader (`build_bddk_monthly_kobi_facts()`) that reads the "Toplam
  Nakdi Krediler" column directly. `load_table_long()` now logs a
  WARNING whenever this fallback triggers.
- **BDDK Weekly (1 series):** weekly new housing-loan originations
  (flow, tablo_289 - see below, summed to monthly).
- **BDDK FinTurk (1 series):** total cash loans, Turkey-wide (sum of
  82 provinces/abroad, forward-filled from quarterly to monthly).

Also, **`gold_health_life_expectancy`** (new, see below) is NOT part of
`gold_fact_long` - it lives in its own table at its native annual
resolution, same reasoning as the FinTurk regional table (forcing it
onto the monthly axis would mean inventing data).

### `gold_bddk_net_profit_monthly`
Phase A2's concrete output: the "PERIOD NET PROFIT (LOSS)" line inside
BDDK Monthly `tablo_2`, **detected as cumulative from the real data**
(`src/gold/scan_cumulative.py`, see `cumulative_scan_report.csv`) and
converted into a true monthly flow. `Value_Cumulative` = the original
YTD value, `Value_Monthly` = the real profit/loss for that month.

### `gold_konut_demo`
A wide table built end-to-end for the hackathon brief's own worked
example (housing credit + interest rate + price index + CPI
deflation). `Konut_Kredisi_Reel_Milyon_TL` = deflated to the 2021-01
base using CPI. **Unit is `milyon TL`** (million TRY - see the "Unit
fix" note below - the column name was corrected accordingly to
`_Milyon_TL`).

**A notable finding from the real data:** nominal housing credit grew
~2.9x between 2021-01 and 2026-06 (276,785 → 801,438 million TRY), but
the inflation-adjusted figure over the same period shrank to about a
third (276,785 → 99,416 million TRY "at 2021-01 prices"). This is a
real, ready-made story for the Causality Tool's showcase example.

### `gold_finturk_regional_quarterly` (new)
BDDK FinTurk `tablo_1` (Loans), **by province**, kept at its own native
quarterly resolution - not forced onto the monthly axis (81 provinces
× monthly forward-fill would triple the row count without adding any
real information). The first regional (`Bolge`) dimension in the gold
layer. Columns: `Period` (`YYYY-Q`, Q∈{3,6,9,12}), `Bolge` (province
name), `Toplam_Nakdi_Kredi_Bin_TL`, `Takipteki_Alacak_Bin_TL`,
`NPL_Orani_Yaklasik`. 1804 rows (82 regions × 22 quarters). Cross-
checked against the national total: `gold_fact_long`'s
`tablo_1.toplam_nakdi_kredi_ulusal` == this table's sum across the 82
provinces (confirmed for 2021-Q1, exact match).

### `gold_health_life_expectancy` (new)
Answers the hackathon's "not just finance" requirement: TUIK's SDMX
API has no dataflow matching "health"/"hospital" (all 427 dataflows
were scanned, same method as the housing/real-estate scan), but
`DF_DNG_YS` ("Life Expectancy at Birth") is real and works. Kept at
its native **annual** resolution, in its own table (forcing it to
monthly would mean inventing data - same reasoning as the FinTurk
regional table). Columns: `Period` (`YYYY`), `Bolge` (`TR`), `Deger`
(years), `Birim` (`Yil`, i.e. "Years"). 4 rows: 2021-2024. **No
2025/2026** - TUIK hasn't published those yet; this is a real gap, not
filled in.

A second candidate (`DF_DOGUM_KDH_C`, province-level crude birth rate)
was investigated but not added: it's part of a shared 19-dimension
birth-statistics DSD, and even after resolving every dimension's "_T"
(Total) code from the DSD's own codelists, the query still returned
"NoRecordsFound" - most likely `YAYIM_DONEMI` (a 528-value
release-period dimension with no total code) needs a specific real
value. Noted as a follow-up in `config/tuik_series.py`, not guessed at.

## Finding resolved — the weekly "Konut" (Housing) line is a flow, not a stock

BDDK Weekly Bulletin `tablo_289` ("Loans") / "a) Konut" (Housing) line
didn't reconcile with BDDK Monthly `tablo_4` / "Consumer Loans -
Housing" for the same month. The cause was found: the weekly figure
moves flat/choppy week to week (38091 → 37319 → 37621 → 38075 → 39452
...) instead of growing smoothly like a monthly stock does - this is
the signature of newly originated housing loans **that week** (a flow),
not a balance. Fixed in `build_gold.py` with `Tip='flow'` and the label
clarified accordingly. The exact official definition could be confirmed
from BDDK's own table documentation/Slack, but the behavioral evidence
is strong.

## Unit fix — BDDK Monthly's real names and units were found

`BDDK_MONTHLY_TABLES` was added to `config/settings.py`: the official
name and unit of each of BDDK Monthly's 17 tables, extracted **without
guessing** directly from the bronze JSON's own `Json.caption` field
(e.g. `data/bronze/bddk_monthly/2021/01/tablo_4_p1.json` →
`"Tuketici Kredileri (milyon TL), Donem:2021/1"`). This means tablo_1-17
can now be referred to by their real names, not just their numbers
(Balance Sheet, Profit & Loss, Loans, Consumer Loans, Sectoral Loan
Distribution, SME Loans, Syndication/Securitization Loans, Securities,
Deposits by Type, Deposits by Maturity, Liquidity Position, Capital
Adequacy, Foreign Currency Position, Off-Balance-Sheet Transactions,
Ratios, Other Information, Foreign Branch Ratios).

**A real bug was found and fixed while doing this:** `tablo_4`'s real
unit is **"million TRY"**; the earlier gold code had mislabeled it as
"thousand TRY". The `Deger` number in `gold_fact_long` never changed -
only the `Birim` label and the column names in `gold_konut_demo`
(`Konut_Kredisi_Bin_TL` → `Konut_Kredisi_Milyon_TL`) were corrected.
In other words, the number 276,785 was always correct, but calling it
"276,785 thousand TRY" was wrong - it's "276,785 million TRY" (~277
billion TRY).

**Note:** unit confirmation was only possible for BDDK Monthly (only
that source has a caption field). BDDK Weekly's HTML page states no
unit anywhere (checked) - `tablo_289.konut`'s `Birim` was therefore
left `NULL` rather than claiming something like "thousand TRY". BDDK
FinTurk's JSON also has no caption; the "thousand TRY" label on
`tablo_1.toplam_nakdi_kredi_ulusal` and
`gold_finturk_regional_quarterly` is an **unconfirmed** assumption
inherited from the `BDDK_FINTURK_TABLES` dictionary in
`config/settings.py` - marked as `"Bin TL (dogrulanmadi)"` (thousand
TRY, unconfirmed) in `gold_fact_long`.

## Data deliberately left out — gold price (EVDS)

`TP.ALTINPIYASA.KAP05` (Borsa Istanbul gold closing price, TRY/gram)
was collected and processed into silver but **not included in gold**:
only 207 irregular observations (gaps up to 28 days), and **no data at
all in 2026** (last observation 2025-11-24). Forcing this onto the
monthly axis (forward-fill) would mean inventing data. The hackathon
brief's own demo scenario already expects this need to be met by
fetching live from Borsa Istanbul's own page via the Web URL Agent
Tool (slide 13) - that's the more accurate source anyway.

## Known limitation — cumulative detection on ratio (%) tables

`scan_cumulative.py` also flagged 9 rows inside `tablo_15` (the
ratio/percentage table) as "cumulative", but these are average-based
ratios - a simple difference (`decumulate()`) doesn't turn a percentage
into a meaningful monthly percentage. Because of this, `build_gold.py`
currently only runs decumulation on `tablo_2` (absolute TRY amounts);
ratio tables are deliberately excluded.

## Cumulative scanning now runs across all 3 sources

`scan_cumulative.py` used to scan only BDDK Monthly; it now also scans
Weekly (114 items, via `run_weekly()`, converting `donemId` to a real
date) and FinTurk (60 items, via `run_finturk()`, on the national-total
basis, excluding ratio columns). Result: **no cumulative items were
found in either** (`cumulative_scan_report_bddk_weekly.csv`,
`cumulative_scan_report_bddk_finturk.csv`) - neither has a
YTD-cumulative income-statement-style pattern like BDDK Monthly does.

## Data-freshness check (new) — `src/gold/check_freshness.py`

Wired into the `agent_controller.py` pipeline (runs after the gold
build). Checks two things: (1) whether `gold_fact_long`'s latest period
actually reaches `FULL_END_YEAR/MONTH` (2026-06) - if not, the build
most likely ran against an incomplete/partial silver layer; (2) whether
any silver file is older than 30 days - an early warning that a source
may need re-fetching, not an error.

## Next steps (rest of Phase A)

1. BDDK Monthly's remaining 13 tables (now 12 after the additions
   above), BDDK Weekly's 8 tables, and FinTurk's 6 tables (Deposits,
   Retail Banking, Selected Sectoral Loans, Ratios, Branches/
   Population, Gold) still haven't been pulled into `gold_fact_long` at
   all - the schema/pattern is ready (`build_bddk_monthly_item_facts()`),
   adding a row is cheap, but which of these are actually needed is a
   product decision.
2. `DF_DOGUM_KDH_C` (province-level crude birth rate) - if the last
   piece of the 19-dimension key (`YAYIM_DONEMI`) can be resolved, this
   becomes a second health series, and a province-level one at that.
3. For adding a new EVDS/TUIK series: NEVER guess a code - first run
   `python -m src.collectors.evds --discover --search <term>` (EVDS) or
   follow the dataflow→DSD→codelist chain (TUIK, see
   `config/tuik_series.py`), then add it to the relevant config file.
