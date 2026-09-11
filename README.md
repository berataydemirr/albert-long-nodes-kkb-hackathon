# KKB data agent

This repo turns BDDK, EVDS and TUIK data into a small, queryable DuckDB
database, then lets an MIA model answer questions about it. The model plans;
Python runs the tools and keeps the data work local.

## Run it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# add MIA_API_KEY to .env
.venv/bin/python -m src.gold.build_gold
.venv/bin/uvicorn web_app:app --reload
```

Open <http://127.0.0.1:8000>. The browser page has a few ready-made prompts,
shows the route chosen, and lists every function call with its arguments and a
short result preview. The right panel can be hidden or reduced to calls only.

You can also use the CLI:

```bash
.venv/bin/python agent.py "Konut kredisi faizi ile hacmi arasındaki ilişki nedir?"
```

## What is available

The router exposes all registered tools for every safe request, so a single
answer can combine them:

- `lakehouse_schema`, `lakehouse_query` — inspect and query the read-only gold database
- `anomaly_analysis`, `change_detection` — find unusual points and sharp moves
- `correlation_analysis`, `causality_analysis`, `threshold_filter_compare` — compare series
- `web_search` — search current public information
- `web_url_extract` — read bounded HTML, PDF, Excel, CSV, text or image URLs

For example, a current-rate question may call `web_search`, open a result with
`web_url_extract`, query the local series, and then run an analysis tool. Unsafe
SQL or prompt-injection requests are refused before the model gets any tools.

Web search uses DuckDuckGo by default. To use Tavily instead, set
`WEB_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` in `.env`.

## Data layout

- `data/bronze/` — source responses
- `data/silver/` — cleaned CSVs
- `data/gold/lakehouse.duckdb` — analysis-ready tables (rebuilt locally)
- `src/collectors/` — source downloads
- `src/processing/` — bronze → silver conversion
- `src/gold/` — silver → DuckDB build
- `src/agent/` — router, model loop and tools
- `docs/` — schema and routing notes

The gold build is network-free; it reads the checked-in silver files. The
collectors need the relevant source keys in `.env` and should be run with care.

## Checks

```bash
.venv/bin/python -m pytest -q
```

Tool calls and model rounds are written to `logs/`. Secrets, generated logs,
the virtual environment and the DuckDB file are ignored by Git.
