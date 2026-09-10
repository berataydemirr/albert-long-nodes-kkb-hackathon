# KKB Agentic AI Hackathon — Data Collection Layer

A layered (bronze/silver/gold) data pipeline collecting and cleaning data
from BDDK (Banking Regulation and Supervision Agency), TCMB EVDS
(Central Bank Electronic Data Distribution System), and TUIK (Turkish
Statistical Institute) for the KKB Data Analytics Hackathon. Bronze and
silver are complete for all 5 sources; a first gold layer (frequency
alignment, cumulative-series correction, a DuckDB build) exists and
covers the housing-credit demo scenario end to end, with most other
BDDK tables still to be added (see "Still pending"). The agentic AI
engine (Cloudeks/LLM integration, tool routing, causality/anomaly/
change-detection tools) is the next phase.

## Quick Start

1. **Setup**: `python -m venv venv` (if needed) → `venv\Scripts\activate`
   → `pip install -r requirements.txt`
2. **Create your `.env`**: copy `.env.example` to `.env` and fill in
   `EVDS_API_KEY` and `TUIK_API_KEY`. `.env` is never committed.
3. **Data is already collected**: cleaned CSVs for BDDK Monthly/Weekly/
   FinTurk, EVDS, and TUIK live under `data/silver/` and are ready to use
   directly with `pandas.read_csv(...)`. See "Directory Map" and
   "Glossary" below for what each source contains.
4. See the "Development Notes" section below for what's done and what's
   still pending.

## Glossary

- **Bronze / Silver / Gold (medallion architecture)**: layered data
  storage. **Bronze** = raw data exactly as received from the source
  (JSON/HTML, untouched). **Silver** = cleaned, columnar, ready-to-use
  tables (CSV). **Gold** = analysis-ready, frequency-aligned dataset,
  built into `data/gold/lakehouse.duckdb` by `python -m src.gold.build_gold`
  (see `docs/gold_schema.md` for the table/column reference and its
  known limitations).
- **Checkpoint / cache**: once a month/table has been downloaded, its
  bronze file's existence is used as a marker to avoid re-downloading it.
- **Endpoint**: a server address that responds to a specific request.
- **jqGrid**: the JSON response shape BDDK uses for its data tables
  (`colNames`, `colModels`, `data: {page, total, records, rows}`).
- **WAF**: Web Application Firewall - bot/automated-request protection.
  A random delay between requests (`waf_sleep()`) exists to avoid
  triggering it.

## Directory Map

```
config/            Shared settings (paths, date range, constants)
src/common/        Shared logging, HTTP session, jqGrid parsing helpers
src/collectors/    Network-facing collectors that write to bronze
src/processing/    Network-free bronze -> silver converters
src/gold/          Network-free silver -> gold: frequency alignment,
                   cumulative-series detection/correction, DuckDB build
docs/              Schema/design notes (e.g. docs/gold_schema.md)
data/bronze/       Raw data (subfolders per source)
data/silver/       Clean, columnar CSVs
data/gold/         lakehouse.duckdb (gitignored, rebuilt by
                   src/gold/build_gold.py) + cumulative_scan_report.csv
data/debug/        Dumps of failed requests / reconnaissance output
logs/              A separate, DEBUG-level log file per run
archive/           Old/exploratory files kept for reference (not on GitHub)
```

## Setup

```
venv\Scripts\activate
pip install -r requirements.txt
```

`EVDS_API_KEY` and `TUIK_API_KEY` must be set in `.env` (see `.env.example`;
`.env` is gitignored and never uploaded).

## Usage

### MIA Agent

The agent layer uses the OpenAI-compatible MIA endpoint from Kloudeks and
keeps execution of analytics tools local against DuckDB. Add `MIA_API_KEY`
to `.env` (never commit the real key), then build the gold database and ask a
question:

```
python -m src.gold.build_gold
python agent.py "2021-2026 arasında konut kredileri ile faiz oranları nasıl değişti?"
```

The first agent iteration exposes read-only Lakehouse queries, robust anomaly
detection, period-over-period change detection, correlation analysis, and a
citation-ready web search. A correlation is explicitly reported as correlation,
not causality. Web URL/PDF and causality tools are planned next. Web search
uses DuckDuckGo by default with no key; set `WEB_SEARCH_PROVIDER=tavily` and
`TAVILY_API_KEY` in `.env` to use Tavily instead.

```
# Single collector, safe/fast test mode (2021-01, tables 1-3):
python -m src.collectors.bddk_monthly --mode test
python -m src.processing.bddk_monthly_silver --mode test

# A small date range:
python -m src.collectors.bddk_monthly --mode range --start-year 2021 --start-month 1 --end-year 2021 --end-month 2 --tables 1,2,3

# Full 2021-2026 scan (can take HOURS, use with care):
python -m src.collectors.bddk_monthly --mode full

# Top-level orchestrator (fetch + parse in sequence, retries on error):
python agent_controller.py --mode test

# EVDS series discovery (the code never guesses series codes):
python -m src.collectors.evds --discover --search konut

# BDDK Weekly Bulletin (donemId = a single increasing week counter, 367=2021-01-08):
python -m src.collectors.bddk_weekly --mode test
python -m src.processing.bddk_weekly_silver --mode test
python -m src.collectors.bddk_weekly --mode range --start-donem 367 --end-donem 370 --tables 289,290

# BDDK FinTurk (by province, quarterly - "HEPSI" returns all provinces in one request):
python -m src.collectors.bddk_finturk --mode test
python -m src.processing.bddk_finturk_silver --mode test
python -m src.collectors.bddk_finturk --mode range --start-year 2021 --start-q 3 --end-year 2021 --end-q 12 --tables 1,2

# TUIK (SDMX API - series registered in config/tuik_series.py):
python -m src.collectors.tuik
python -m src.processing.tuik_silver

# Gold layer (network-free, reads only from data/silver/):
python -m src.gold.scan_cumulative   # which BDDK Monthly line items are YTD-cumulative (writes a report)
python -m src.gold.build_gold        # builds data/gold/lakehouse.duckdb
```

## Known Quirks

- **BDDK cumulative data**: some BDDK Monthly tables are year-to-date
  cumulative, others aren't. `src/gold/scan_cumulative.py` detects this
  from the data itself (monotonic-within-year + resets at year
  boundaries) instead of guessing per table - confirmed `tablo_2`
  (income statement) is cumulative; `src/gold/cumulative.py` converts it
  to a true monthly flow. Ratio/percentage tables (e.g. `tablo_15`) are
  deliberately excluded from auto-correction - diffing a percentage
  isn't meaningful. BDDK Weekly and FinTurk haven't been scanned for
  cumulative items yet.
- **`session.verify=False`**: deliberate. bddk.org.tr genuinely serves a
  broken SSL certificate chain (independently confirmed). Not a security
  hole - see the comment in `src/common/http.py`.
- **CSV encoding: always pass `encoding="utf-8"` explicitly**, both
  reading and writing. Without it, `pandas.read_csv`/`to_csv` fall back
  to the OS locale encoding - on a Windows machine with a Turkish
  locale (cp1254) this silently corrupts Turkish characters (`İ`, `Ö`,
  `Ş`...) into `�`, even though the files on disk are valid UTF-8. Every
  script in `src/processing/` and `src/gold/` already does this; any new
  code touching these CSVs must too, or Turkish text can render broken
  live in front of the jury depending on which machine runs it. (Plain
  terminal/log output showing `�` for a correctly-encoded string, e.g.
  `İSTANBUL` as `�STANBUL`, is a separate, harmless issue - only the
  console's code page, not the data; `chcp 65001` fixes the display.)
- **BDDK Weekly Bulletin uses a completely different architecture** from
  the Monthly Bulletin: not a JSON API, but a session-based, multi-step
  form flow, and the server's anti-forgery check also validates the
  `Origin`/`Referer` headers (missing them silently returns 200 without
  changing anything - no error). See
  `archive/exploration/recon_bddk_haftalik_findings.md` for details
  (local reference only, not part of the GitHub upload).
- **TCMB EVDS caps daily-frequency series at ~1000 rows per request** and
  silently truncates a longer requested range instead of erroring.
  Collectors fetch in one-year chunks to stay under this limit.
- **TUIK's SDMX catalog has no housing-related dataflow.** All 427
  dataflows were scanned; none relate to "konut" (housing) or "emlak"
  (real estate). Rely on BDDK/EVDS for housing data instead.

## Development Notes

**Root cause found and fixed**: the original BDDK monthly collector read
the API response from the wrong JSON path (`Json.rows`/`Json.total`
instead of the real `Json.data.rows`/`Json.data.total`). Because
`dict.get()` silently falls back to a default instead of raising, this
never crashed - it just always produced zero rows, so the silver layer
was completely empty. The fix lives in `src/common/jqgrid.py`, shared by
the monthly and FinTurk collectors.

**What's working, verified against live re-fetches independent of the
collector code**:
- BDDK Monthly Bulletin - full 2021-01..2026-06 scan complete (1122/1122
  files, 0 missing, 0 empty, consistent row counts per table across all
  66 months).
- BDDK Weekly Bulletin - required reverse-engineering a session-based
  multi-step form flow (not a JSON API); root cause of an early failure
  was missing `Origin`/`Referer` headers, found by comparing a captured
  real-browser request (via Playwright) against our own request.
- BDDK FinTurk (by province) - the `"HEPSİ"` (ALL) query returns every
  province in a single request, which reduces a full scan from tens of
  thousands of requests to about 150.
- TCMB EVDS - 11 series covering house prices, interest rates (housing/
  consumer loan/deposit/policy), inflation, FX, M2 money supply, card
  spending, industrial production, and business confidence. All found
  via `--discover` and confirmed with `get_series()` - no code guessed.
  A 12th (gold price) was found and collected but deliberately excluded
  from the gold layer for data-quality reasons - see `docs/gold_schema.md`.
- TUIK SDMX API - authentication flow implemented per the official docs;
  one labor-force/unemployment series added (no housing data available
  in this catalog, confirmed).
- A background full backfill (2021-2026, all BDDK sources) ran to
  completion with zero errors logged.
- **Gold layer (first pass)**: frequency alignment (daily/weekly/
  quarterly -> monthly), cumulative-series detection+correction, and a
  DuckDB build (`data/gold/lakehouse.duckdb`) - see `docs/gold_schema.md`
  for the full table reference, what's in it, and what's deliberately
  left out. Currently covers: all 11 EVDS series, TUIK unemployment,
  BDDK Monthly's housing-loan line, BDDK Weekly's housing-loan line
  (reclassified from stock to flow after the data showed it doesn't
  behave like a balance), and BDDK FinTurk's province-level loan data
  (both a national total and the first regional dimension in the gold
  layer). Verified end to end: the housing-credit demo scenario from
  the hackathon brief's own worked example (slides 11-12) reproduces
  correctly, including CPI-deflated ("enflasyondan arındırılmış")
  values.

**Still pending** (see `docs/gold_schema.md` "Sıradaki adımlar" for the
full list):
- Most BDDK Monthly (16 of 17 tables), BDDK Weekly (8 of 9 tables), and
  BDDK FinTurk (6 of 7 tables) line items haven't been pulled into the
  gold layer yet - only the housing-credit theme has been processed
  end to end so far. (Every BDDK Monthly table now has a real name/unit
  in `BDDK_MONTHLY_TABLES`, pulled from the bronze JSON's own `caption`
  field rather than guessed - the remaining work is wiring specific
  line items into `gold_fact_long`, not naming them.)
- Cumulative-series scanning (`scan_cumulative.py`) has only run against
  BDDK Monthly; BDDK Weekly and FinTurk haven't been checked.
- Test coverage is limited to the network-free pure functions in
  `src/gold/` (`pytest tests/ -v`, also run in CI) - the collectors,
  silver converters and `build_gold.py`'s own I/O aren't covered.
- Health-data source (to demonstrate the "not just finance" requirement)
  - not yet researched.
- The agentic engine itself (Cloudeks/LLM integration, tool routing,
  causality/anomaly/change-detection tools) - next phase.
