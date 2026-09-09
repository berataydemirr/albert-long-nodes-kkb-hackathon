# KKB Agentic AI Hackathon — Data Collection Layer

A layered (bronze/silver/gold) data pipeline collecting and cleaning data
from BDDK (Banking Regulation and Supervision Agency), TCMB EVDS
(Central Bank Electronic Data Distribution System), and TUIK (Turkish
Statistical Institute) for the KKB Data Analytics Hackathon. The agentic
AI engine (causality/anomaly/change-detection tools) is the next phase;
this repository currently focuses on reliable data collection.

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
  tables (CSV). **Gold** = analysis-ready, frequency-aligned final
  dataset (not built yet).
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
data/bronze/       Raw data (subfolders per source)
data/silver/       Clean, columnar CSVs
data/gold/         (empty - not built yet)
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
```

## Known Quirks

- **BDDK cumulative data**: some BDDK tables are cumulative since their
  publication start date, others reset to cumulative-per-year. This is
  not yet resolved at the silver layer - it belongs in the gold layer,
  alongside frequency alignment.
- **`session.verify=False`**: deliberate. bddk.org.tr genuinely serves a
  broken SSL certificate chain (independently confirmed). Not a security
  hole - see the comment in `src/common/http.py`.
- **Turkish characters may look garbled in a Windows terminal** (e.g.
  `Sekt?r`): this is only the terminal's code page, the CSV files
  themselves are valid UTF-8. `chcp 65001` fixes the display; opening in
  Excel/pandas already shows it correctly.
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
- TCMB EVDS - 6 series covering house prices, interest rates, inflation,
  FX, and industrial production, chosen to match the hackathon's own
  causality example (interest rates vs. house prices).
- TUIK SDMX API - authentication flow implemented per the official docs;
  one labor-force/unemployment series added (no housing data available
  in this catalog, confirmed).
- A background full backfill (2021-2026, all BDDK sources) ran to
  completion with zero errors logged.

**Still pending**:
- Gold layer: frequency alignment across sources (monthly/weekly/
  quarterly/daily) and resolving the cumulative-data quirk above.
- Health-data source (to demonstrate the "not just finance" requirement)
  - not yet researched.
- The agentic engine itself (Cloudeks/LLM integration, tool routing,
  causality/anomaly/change-detection tools) - next phase.
