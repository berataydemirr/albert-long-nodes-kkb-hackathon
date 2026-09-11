# KKB Agentic AI Hackathon — Data Collection Layer

A layered (bronze/silver/gold) data pipeline collecting and cleaning data
from BDDK (Banking Regulation and Supervision Agency), TCMB EVDS
(Central Bank Electronic Data Distribution System), and TUIK (Turkish
Statistical Institute) for the KKB Data Analytics Hackathon. Bronze and
silver are complete for all 5 sources; a first gold layer (frequency
alignment, cumulative-series correction, a DuckDB build) exists and
covers the housing-credit demo scenario end to end, with most other
BDDK tables still to be added (see "Still pending"). The agentic AI
engine (function-calling LLM, tool routing, causality/anomaly/
change-detection tools) is complete — see the `agentic-harness-berat`
branch.

## Quick Start

1. **Setup**: `python -m venv venv` (if needed) → `venv\Scripts\activate`
   → `pip install -r requirements.txt`
2. **Create your `.env`**: copy `.env.example` to `.env` and fill in
   `EVDS_API_KEY`, `TUIK_API_KEY`, and `MIA_API_KEY`. `.env` is never committed.
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
src/agent/         MIA agent: orchestrator, tools, client
src/report/        Citation extraction, Plotly charts, HTML export
docs/              Schema/design notes (e.g. docs/gold_schema.md)
data/bronze/       Raw data (subfolders per source)
data/silver/       Clean, columnar CSVs
data/gold/         lakehouse.duckdb (gitignored, rebuilt by
                   src/gold/build_gold.py) + cumulative_scan_report.csv
data/debug/        Dumps of failed requests / reconnaissance output
logs/              A separate, DEBUG-level log file per run
benchmarks/        Agent quality benchmark suite
dashboard/         Streamlit dashboard (Overview + Ask tabs)
archive/           Old/exploratory files kept for reference (not on GitHub)
```

## Setup

```
venv\Scripts\activate
pip install -r requirements.txt
```

`EVDS_API_KEY` and `TUIK_API_KEY` must be set in `.env` (see `.env.example`;
`.env` is gitignored and never uploaded).

## MIA Agent

Add `MIA_API_KEY` to `.env` (see `.env.example`), then:

```
python -m src.gold.build_gold
python agent.py "How did the housing credit interest rate change between 2021 and 2026?"
```

Tools (`src/agent/tools.py`): `lakehouse_schema`, `lakehouse_query`
(guardrail: single SELECT only, known tables only, automatic LIMIT),
`anomaly_analysis`, `change_detection`, `correlation_analysis`,
`causality_analysis` (lag correlation + Granger test),
`threshold_filter_compare`. All calls are logged to
`logs/agent_tools_*.log` and `logs/orchestrator_*.log`.

### Benchmark

```
python -m benchmarks.agent_benchmark
```

Calls the real MIA API (does not run in CI). Measures 6 metrics:

| Metric | What it measures |
|---|---|
| `tool_selection_accuracy` | Was the expected tool actually called? |
| `hallucination_rate` | Did the SQL reference a Seri_Kod that does not exist in the gold layer? |
| `guardrail_robustness` | Could a prompt-injection attempt execute DROP/DELETE? |
| `consistency` | Does the same question always select the same tool set? |
| `avg_latency_sec` | Average round-trip time per question |
| `source_citation_rate` | Does the answer contain a concrete period/Seri_Kod reference? |

Results are written to `benchmarks/last_run_report.json`. To compare two
harness or model configurations, run the same question set through both
and compare these 6 numbers.

The tools are also stress-tested with adversarial edge cases
(`tests/test_agent_tools_stress.py`,
`pytest tests/test_agent_tools_stress.py -v`): SQL injection variants
(system table reads, file reads, stacked statements, comment smuggling),
NaN/zero-variance inputs, duplicate-period merge blowups.
6 real bugs were found and fixed — see the commit message for details.

## Report & Dashboard

```
pip install -r requirements.txt   # streamlit + plotly included
streamlit run dashboard/app.py
```

Two tabs:

- **Ask**: natural-language question → MIA agent → answer + source
  citations (`src/report/citations.py`: which Seri_Kod/table was used,
  did any tool call fail, did the model answer without calling any tool —
  "ungrounded" answers are explicitly flagged) + auto-generated Plotly
  chart for relevant series + HTML report download (`src/report/export.py`,
  self-contained single file).
- **Overview**: static KPI panel that works without a MIA API key —
  housing credit nominal/real, policy rate → housing credit causality test
  (lag chart + Granger p-values), anomaly scan on any selectable series,
  full series catalog.

`src/report/` is covered by `tests/test_report.py` (27 tests): empty/
sparse data, HTML injection (a `<script>` tag in a model answer is not
injected into the report — `html.escape()` sanitises it), single-point
series, mismatched lengths — none crash, all degrade gracefully to a
"no data" placeholder.

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

### Browser UI

With `MIA_API_KEY` configured in `.env`, run the local UI and open the URL
printed by Uvicorn (normally `http://127.0.0.1:8000`):

```
uvicorn web_app:app --reload
```

Ask a current-information question such as “TCMB'nin güncel politika faizi
nedir? Kaynaklarıyla ver.” The agent can select `web_search`; inspect the
source URLs it returns. The UI is local-only and does not expose API keys.

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
python -m src.gold.scan_cumulative   # detect YTD-cumulative line items (writes 3 reports)
python -m src.gold.build_gold        # builds data/gold/lakehouse.duckdb
python -m src.gold.check_freshness   # verify gold coverage reaches the target end date
```

## Known Quirks

- **BDDK cumulative data**: some BDDK Monthly tables are year-to-date
  cumulative, others aren't. `src/gold/scan_cumulative.py` detects this
  from the data itself (monotonic-within-year + resets at year
  boundaries) instead of guessing per table — confirmed `tablo_2`
  (income statement) is cumulative; `src/gold/cumulative.py` converts it
  to a true monthly flow. Ratio/percentage tables (e.g. `tablo_15`) are
  deliberately excluded from auto-correction — diffing a percentage
  isn't meaningful. BDDK Weekly and FinTurk haven't been scanned for
  cumulative items yet.
- **`session.verify=False`**: deliberate. bddk.org.tr genuinely serves a
  broken SSL certificate chain (independently confirmed). Not a security
  hole — see the comment in `src/common/http.py`.
- **CSV encoding: always pass `encoding="utf-8"` explicitly**, both
  reading and writing. Without it, `pandas.read_csv`/`to_csv` fall back
  to the OS locale encoding — on a Windows machine with a Turkish
  locale (cp1254) this silently corrupts Turkish characters (`İ`, `Ö`,
  `Ş`...) into `<?>`, even though the files on disk are valid UTF-8. Every
  script in `src/processing/` and `src/gold/` already does this; any new
  code touching these CSVs must too. (Plain terminal/log output showing
  garbled characters, e.g. `ISTANBUL` as `?STANBUL`, is a separate,
  harmless issue — only the console's code page, not the data;
  `chcp 65001` fixes the display.)
- **BDDK Weekly Bulletin uses a completely different architecture** from
  the Monthly Bulletin: not a JSON API, but a session-based, multi-step
  form flow, and the server's anti-forgery check also validates the
  `Origin`/`Referer` headers (missing them silently returns 200 without
  changing anything — no error). See
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
never crashed — it just always produced zero rows, so the silver layer
was completely empty. The fix lives in `src/common/jqgrid.py`, shared by
the monthly and FinTurk collectors.

**What's working, verified against live re-fetches independent of the
collector code**:
- BDDK Monthly Bulletin — full 2021-01..2026-06 scan complete (1122/1122
  files, 0 missing, 0 empty, consistent row counts per table across all
  66 months).
- BDDK Weekly Bulletin — required reverse-engineering a session-based
  multi-step form flow (not a JSON API); root cause of an early failure
  was missing `Origin`/`Referer` headers, found by comparing a captured
  real-browser request (via Playwright) against our own request.
- BDDK FinTurk (by province) — the `"HEPSİ"` (ALL) query returns every
  province in a single request, which reduces a full scan from tens of
  thousands of requests to about 150.
- TCMB EVDS — 11 series covering house prices, interest rates (housing/
  consumer loan/deposit/policy), inflation, FX, M2 money supply, card
  spending, industrial production, and business confidence. All found
  via `--discover` and confirmed with `get_series()` — no code guessed.
  A 12th (gold price) was found and collected but deliberately excluded
  from the gold layer for data-quality reasons — see `docs/gold_schema.md`.
- TUIK SDMX API — authentication flow implemented per the official docs;
  labor-force/unemployment and life-expectancy series added (no housing
  dataflow in this catalog, confirmed; no direct hospital dataflow
  either, but life expectancy satisfies the "not just finance"
  requirement — see `docs/gold_schema.md`).
- A background full backfill (2021-2026, all BDDK sources) ran to
  completion with zero errors logged.
- **Gold layer**: frequency alignment (daily/weekly/quarterly ->
  monthly), cumulative-series detection+correction (now scanned across
  all 3 BDDK sources — Monthly/Weekly/FinTurk, not just Monthly), a
  data-freshness check (`src/gold/check_freshness.py`, wired into
  `agent_controller.py`), and a DuckDB build — see `docs/gold_schema.md`
  for the full table reference. Covers: all 11 EVDS series, TUIK
  unemployment + life expectancy, 4 BDDK Monthly lines (housing credit,
  total loans, total deposits, capital adequacy ratio), BDDK Weekly's
  housing-loan line (reclassified stock -> flow after the data showed
  it doesn't behave like a balance), and BDDK FinTurk's province-level
  loan data (national total + first regional dimension). Verified end
  to end: the housing-credit demo scenario from the hackathon brief's
  own worked example (slides 11-12) reproduces correctly, including
  CPI-deflated values.

**Still pending** (see `docs/gold_schema.md` for the full list):
- Most BDDK Monthly (13 of 17 tables), BDDK Weekly (8 of 9 tables), and
  BDDK FinTurk (6 of 7 tables) line items still haven't been pulled into
  the gold layer — which ones matter is a product decision, not an
  engineering blocker (every table now has a real name/unit in
  `BDDK_MONTHLY_TABLES`, and `build_bddk_monthly_item_facts()` makes
  adding one a few lines of code).
- A second health series (province-level crude birth rate,
  `DF_DOGUM_KDH_C`) was investigated but not resolved — its 19-dimension
  key needs one more dimension's real value, not a guessed one.
- Test coverage is limited to the network-free pure functions in
  `src/gold/` and `src/agent/tools.py` (`pytest tests/ -v`, also run in
  CI) — the collectors, silver converters and `build_gold.py`'s own I/O
  aren't covered.
