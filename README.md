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
- `compare_regions` — rank provinces by lending or NPL ratio, with optional growth
- `web_search` — search current public information
- `web_url_extract` — read bounded HTML, PDF, Excel, CSV, text or image URLs

The analysis tools take a `seri_kod` and read the series from the database
themselves. Passing explicit `periods`/`values` arrays still works, but is only
meant for data that is not in the lakehouse.

`gold_fact_long` is national — every row is `Bolge='TR'`. Province data lives in
`gold_regional_long`: 81 provinces plus `YURT DIŞI`, 22 quarters from 2021-12 to
2026-6, and **35 metrics** — lending, deposits, consumer credit by product
(housing, vehicle, cards, overdraft) with its non-performing part, per-capita
figures, branch counts, and BDDK's own default ratios including per-sector ones.

It is reachable only through `compare_regions`, which returns a ranked head
(capped at 25) rather than all 82 rows. The metric is a *string validated
against the table*, not a schema enum: metrics are rows, so 35 of them cost the
model's context nothing and a metric added in a future gold build is reachable
with no code change. An unknown name comes back with the full list.

Three traps it handles, each of which produces a wrong answer rather than an
error:

- Quarter labels are not zero-padded, so `'2021-12' < '2021-3'` as strings.
  Every ordering goes through a parsed `(year, quarter)` key.
- `YURT DIŞI` is not a province. It is excluded from rankings by default and
  reported separately rather than silently dropped.
- Units are carried verbatim, never rescaled. BDDK ships the same concept at two
  scales — `npl` here is its published percentage (1.56–9.29), while the derived
  column in the older wide table is a 0–1 fraction. Shares and totals are
  reported only for additive metrics, because a "share of an NPL ratio" means
  nothing.

For example, a current-rate question may call `web_search`, open a result with
`web_url_extract`, query the local series, and then run an analysis tool. Unsafe
SQL or prompt-injection requests are refused before the model gets any tools.

Web search defaults to DuckDuckGo, but **DuckDuckGo's HTML endpoint answers
with a bot challenge from some networks**, including the one this was
developed on ("select all squares containing a duck"), so it may return
nothing at all. This deployment runs on Tavily instead
(`WEB_SEARCH_PROVIDER=tavily` + `TAVILY_API_KEY` in `.env`) — set your own key
the same way if DuckDuckGo is blocked from your network too.

A blocked provider says so rather than returning an empty result list: a
blocked DuckDuckGo comes back with `provider_available: false` and an
instruction not to retry, instead of a result shape that reads as "your query
found nothing". That distinction matters more than it sounds — with the old
behaviour the model rephrased and searched four times in five rounds against a
provider that could never answer (32.4 s, 6 tool calls). The same question
against Tavily now answers correctly, citing TCMB's own announcement, in
8.8 s with one search call and zero tool errors.

A second guard sits in the tool loop for the general case: an identical repeat
call is refused, and any single tool is cut off after three calls in one
answer. Both come back as normal tool results explaining what to do instead, so
the model changes strategy rather than being cut off mid-conversation.

## Data layout

- `data/bronze/` — source responses
- `data/silver/` — cleaned CSVs
- `data/gold/lakehouse.duckdb` — analysis-ready tables (rebuilt locally)
- `src/collectors/` — source downloads
- `src/processing/` — bronze → silver conversion
- `src/gold/` — silver → DuckDB build
- `src/agent/` — router, model loop and tools
- `docs/` — schema, routing and [agent architecture](docs/agent_architecture.md)

The gold build is network-free; it reads the checked-in silver files. The
collectors need the relevant source keys in `.env` and should be run with care.

## Cost and speed

Wall time is almost entirely the model, not the data layer:

```bash
.venv/bin/python -m benchmarks.context_cost_benchmark       # local, no API calls
.venv/bin/python -m benchmarks.reasoning_effort_benchmark   # hits the API
```

| What | Result |
|---|---|
| `lakehouse_schema` / `lakehouse_query` / `resolve_series` | 15–31 ms each |
| Tool time in a real run | 0.66 s of 32.4 s (2%) |
| `causality_analysis` arguments, array path vs `seri_kod` | 2372 → 80 chars (29.6x), identical results |
| History over a simulated 8-round chain, after compaction | 5137 → 2282 est. tokens (55.6%), no message dropped |
| Reasoning share of one long answer | 18,212 of 19,568 completion tokens (93%) |
| One causality question, before and after the changes below | 32.4 s → 7.9 s |

The model is a reasoning model, so its time goes to reasoning tokens rather
than to the answer or to us. Four changes follow from that:

- The series catalog is injected into the system prompt, so the model no longer
  opens with a `lakehouse_schema` round just to learn the codes.
- The analysis tools read series by `seri_kod` instead of having the model
  retype the values.
- The prompt bounds answer length and forbids repeating a tool call, because
  every extra round and every generated token is wall time.
- `MIA_REASONING_EFFORT` (default `low`) controls how much the model thinks.
  Measured across levels, tool selection and source citation held at 1.00
  throughout, so the cheaper levels are not trading correctness for speed;
  `none` is faster still if you want it.

Streaming was measured and deliberately not built: the model emits its answer
only after reasoning finishes, so 96 s of a 102 s response is a blank screen
either way — streaming would have exposed 6% of the wait.

Older tool outputs are replaced with short digests once history passes a
budget. That is a cost and context-limit guard rather than a speed one;
compaction only rewrites message content and never drops a message, because a
`tool` message that loses its pairing breaks the next API call.

Every run reports where its time and context went (`return_timing=True`, also
shown in the browser page), so a slow answer can be attributed rather than
guessed at.

## Checks

```bash
.venv/bin/python -m pytest -q
```

Tool calls and model rounds are written to `logs/`. Secrets, generated logs,
the virtual environment and the DuckDB file are ignored by Git.
