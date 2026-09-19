"""
Project-wide paths, date range, and constants. Single source of truth -
every collector/processing script imports this instead of redefining
the same values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- .env is loaded once, here ---
load_dotenv(PROJECT_ROOT / ".env")
EVDS_API_KEY = os.getenv("EVDS_API_KEY")
TUIK_API_KEY = os.getenv("TUIK_API_KEY")
MIA_API_KEY = os.getenv("MIA_API_KEY")
MIA_BASE_URL = os.getenv("MIA_BASE_URL", "https://mia.csp.kloudeks.com/v1")
MIA_CHAT_MODEL = os.getenv("MIA_CHAT_MODEL", "kkbhackathon2026/Qwen3.8-27B")
MIA_EMBEDDING_MODEL = os.getenv("MIA_EMBEDDING_MODEL", "kkbhackathon2026/Qwen3-Embedding-8B")

# The chat model is a reasoning model, and reasoning is where its wall time
# actually goes - measured at 18,212 reasoning tokens out of 19,568 for one
# long answer (93%). The endpoint accepts: none, minimal, low, medium, high,
# xhigh, max. Set per deployment; see benchmarks/reasoning_effort_benchmark.py
# for the measured speed/quality trade-off behind the default.
MIA_REASONING_EFFORT = os.getenv("MIA_REASONING_EFFORT", "low")
# Unset, the router picks the effort per route (a lookup runs lower than a
# causality chain). Set explicitly, the env value wins for every route.
MIA_REASONING_EFFORT_EXPLICIT = "MIA_REASONING_EFFORT" in os.environ
# Wall-clock budget for one ask() run, alongside the round limit. A single
# model call may take up to 240s, so rounds alone do not bound the worst case.
# Raised from 300 to 600: live evidence on a broad multi-table discovery
# question showed individual rounds regularly taking 200-250s+ under normal
# shared-endpoint load - at 300s, a route with max_rounds=8 (e.g. ANALYSIS)
# could never actually reach more than ~1 round before the deadline cut it
# off, making the round budget meaningless. Safe to raise now that two
# earlier fixes exist: web_app.py streams heartbeats on /api/ask so no
# proxy/tunnel truncates a long-running request regardless of duration, and
# _call_model's own retries are deadline-aware so they never waste time
# past whatever budget is set here.
AGENT_RUN_DEADLINE_SEC = float(os.getenv("AGENT_RUN_DEADLINE_SEC", "600"))
# Extra model call when an answer states numbers no tool returned. Off by
# default: the heuristic flags legitimate arithmetic (a sum of two shares)
# and one repair call measured 50-85s on the shared endpoint. The check
# itself always runs and is reported in timing.summary.grounding.
AGENT_GROUNDING_REPAIR = os.getenv("AGENT_GROUNDING_REPAIR", "0") == "1"
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
# Self-hosted SearXNG instance (see docker-compose.searxng.yml). Open-source
# requirement: no third-party paid search API (Tavily was removed for this
# reason - see git history if it needs to come back).
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")

# --- Data layers (medallion architecture: bronze=raw, silver=clean, gold=analysis-ready) ---
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DEBUG_DIR = DATA_DIR / "debug"
# User-uploaded CSV/Excel/PDF files (see src/agent/data_import.py) land
# directly in lakehouse.duckdb, in their own table named after the
# dataset - so a single lakehouse_query call can JOIN a freshly-imported
# table against the curated gold_* tables, per the team's explicit ask
# that imported/fetched data be queryable alongside the vetted layer, not
# walled off in a second file. Each imported table is still tracked in
# its own registry table (_import_registry) and reported as
# user-supplied/unverified wherever it's surfaced (see lakehouse_schema),
# and dataset_name can never collide with a real gold_* table - see
# data_import.py's own docstring for the full reasoning and history.
USER_UPLOADS_DIR = DATA_DIR / "uploads"
LAKEHOUSE_DB_PATH = GOLD_DIR / "lakehouse.duckdb"

# Opt-in HTTP Basic Auth for web_app.py, gated by both vars being set -
# normal local dev (nothing set) is unaffected. Used when the server is
# exposed publicly (e.g. a tunnel for a one-off demo) so a stranger who
# finds the URL can't burn the MIA API quota or upload files.
DEMO_AUTH_USER = os.getenv("DEMO_AUTH_USER")
DEMO_AUTH_PASS = os.getenv("DEMO_AUTH_PASS")
LOGS_DIR = PROJECT_ROOT / "logs"

BDDK_MONTHLY_BRONZE_DIR = BRONZE_DIR / "bddk_monthly"
BDDK_MONTHLY_SILVER_DIR = SILVER_DIR / "bddk_monthly"
BDDK_WEEKLY_BRONZE_DIR = BRONZE_DIR / "bddk_weekly"
BDDK_WEEKLY_SILVER_DIR = SILVER_DIR / "bddk_weekly"
BDDK_FINTURK_BRONZE_DIR = BRONZE_DIR / "bddk_finturk"
BDDK_FINTURK_SILVER_DIR = SILVER_DIR / "bddk_finturk"
EVDS_BRONZE_DIR = BRONZE_DIR / "evds"
EVDS_SILVER_DIR = SILVER_DIR / "evds"
TUIK_BRONZE_DIR = BRONZE_DIR / "tuik"
TUIK_SILVER_DIR = SILVER_DIR / "tuik"

for _d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, DEBUG_DIR, LOGS_DIR,
           BDDK_MONTHLY_BRONZE_DIR, BDDK_MONTHLY_SILVER_DIR,
           BDDK_WEEKLY_BRONZE_DIR, BDDK_WEEKLY_SILVER_DIR,
           BDDK_FINTURK_BRONZE_DIR, BDDK_FINTURK_SILVER_DIR, EVDS_BRONZE_DIR,
           EVDS_SILVER_DIR, TUIK_BRONZE_DIR, TUIK_SILVER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Date range (hackathon brief: 2021-01 .. 2026-06) ---
FULL_START_YEAR, FULL_START_MONTH = 2021, 1
FULL_END_YEAR, FULL_END_MONTH = 2026, 6

# --- BDDK constants ---
BDDK_TABLE_IDS = list(range(1, 18))  # tables 1-17
BDDK_TARAF_SEKTOR = "10001"          # "Sektor" = whole banking sector aggregate (not a single bank) - confirmed
BDDK_PARA_BIRIMI_DEFAULT = "TL"

# BDDK Monthly Bulletin's own table names/units, straight from the
# "caption" field the bronze JSON already carries (e.g.
# data/bronze/bddk_monthly/2021/01/tablo_4_p1.json ->
# Json.caption == "Tuketici Kredileri (milyon TL), Donem:2021/1") -
# not guessed. IMPORTANT: units differ per table (most are "milyon TL",
# tablo_5 is "bin TL", tablo_15/16/17 have none - they're ratios/counts)
# - a table without a name/unit mapping here should NOT be assumed to
# share another table's unit. Weekly/FinTurk tables don't expose a
# caption the same way (checked: FinTurk's JSON has no such field, and
# the Weekly HTML page doesn't state a unit in its visible text either)
# - their units in BDDK_WEEKLY_TABLES/BDDK_FINTURK_TABLES below are
# unconfirmed, unlike this dict.
BDDK_MONTHLY_TABLES = {
    1: {"name": "Bilanço", "unit": "milyon TL"},
    2: {"name": "Kar Zarar", "unit": "milyon TL"},
    3: {"name": "Krediler", "unit": "milyon TL"},
    4: {"name": "Tüketici Kredileri", "unit": "milyon TL"},
    5: {"name": "Sektörel Kredi Dağılımı", "unit": "bin TL"},
    6: {"name": "KOBİ Kredileri", "unit": "milyon TL"},
    7: {"name": "Sendikasyon Seküritizasyon Kredileri", "unit": "milyon TL"},
    8: {"name": "Menkul Kıymetler", "unit": "milyon TL"},
    9: {"name": "Mevduat Türler İtibarıyla", "unit": "milyon TL"},
    10: {"name": "Mevduat Vade İtibarıyla", "unit": "milyon TL"},
    11: {"name": "Likidite Durumu", "unit": "milyon TL"},
    12: {"name": "Sermaye Yeterliliği", "unit": "milyon TL"},
    13: {"name": "Yabancı Para Pozisyonu", "unit": "milyon TL"},
    14: {"name": "Bilanço Dışı İşlemler", "unit": "milyon TL"},
    15: {"name": "Rasyolar", "unit": None},
    16: {"name": "Diğer Bilgiler", "unit": None},
    17: {"name": "Yurt Dışı Şube Rasyoları", "unit": None},
}

# --- WAF-friendly delay (random pause between requests) ---
WAF_SLEEP_MIN_SEC = 1.5
WAF_SLEEP_MAX_SEC = 3.0

# --- Test-mode defaults (safe/fast default: small, single month) ---
TEST_YEAR, TEST_MONTH = 2021, 1
TEST_TABLE_IDS = [1, 2, 3]

# --- BDDK Weekly Bulletin constants ---
# The weekly bulletin addresses a period by a SINGLE increasing counter
# called "donemId" (not year+week). Reference points confirmed via live
# reconnaissance (week 1 of January for each year):
#   2021-01 (week 1) -> donemId=367   2024-01 (week 1) -> donemId=523
#   2022-01 (week 1) -> donemId=419   2025-01 (week 1) -> donemId=575
#   2023-01 (week 1) -> donemId=471   2026-01 (week 1) -> donemId=627
# Last week of June 2026 (week 26) -> donemId=652
# (see archive/exploration/recon_bddk_haftalik_findings.md)
BDDK_WEEKLY_DONEM_START = 367  # 2021-01-08 (week 1)
BDDK_WEEKLY_DONEM_END = 652    # 2026-06-26 (week 26) - the hackathon's upper bound

BDDK_WEEKLY_TABLES = {
    289: "Loans",
    290: "Non-Performing Receivables",
    291: "Securities",
    292: "Deposits",
    293: "Other Balance Sheet Items",
    294: "Off-Balance-Sheet Transactions",
    295: "Securities Held in Custody - 1",
    296: "Securities Held in Custody - 2",
    297: "Foreign Currency Position",
}
BDDK_WEEKLY_TABLE_IDS = list(BDDK_WEEKLY_TABLES.keys())

TEST_DONEM_ID = BDDK_WEEKLY_DONEM_START  # 2021-01-08, week 1
TEST_WEEKLY_TABLE_IDS = [289, 290]

# --- BDDK FinTurk (by province) constants ---
# FinTurk publishes QUARTERLY data - the "donem" field uses "YYYY-Q"
# format (Q: 3, 6, 9, or 12). Selecting the special "HEPSI" (ALL) value
# for the province returns ALL provinces in a SINGLE request (81
# provinces + "abroad" = 82 rows, confirmed live) - no need to query
# province by province.
BDDK_FINTURK_TABLES = {
    1: "Loans (Thousand TRY)",
    2: "Deposits (Thousand TRY)",
    3: "Retail Banking (Thousand TRY)",
    4: "Selected Sectoral Loans (Thousand TRY)",
    5: "Ratios (%)",
    6: "Branches (Count) and Population Distribution",
    7: "Gold Loans and Gold Deposits (Thousand TRY)",
}
BDDK_FINTURK_TABLE_IDS = list(BDDK_FINTURK_TABLES.keys())
BDDK_FINTURK_SEHIR_HEPSI = "HEPSİ"  # literal API value meaning "ALL provinces"
BDDK_FINTURK_QUARTERS = (3, 6, 9, 12)
BDDK_FINTURK_START = (2021, 3)   # first quarter >= 2021-01
BDDK_FINTURK_END = (2026, 6)     # the hackathon's upper bound

TEST_FINTURK_DONEM = "2021-3"
TEST_FINTURK_TABLE_IDS = [1]


def quarter_range(start_year: int, start_q: int, end_year: int, end_q: int):
    """Yields (year, quarter) pairs as 'YYYY-Q' strings from start to end (inclusive)."""
    y, qi = start_year, BDDK_FINTURK_QUARTERS.index(start_q)
    end_key = (end_year, BDDK_FINTURK_QUARTERS.index(end_q))
    while (y, qi) <= end_key:
        yield f"{y}-{BDDK_FINTURK_QUARTERS[qi]}"
        qi += 1
        if qi >= len(BDDK_FINTURK_QUARTERS):
            qi = 0
            y += 1


def month_range(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yields (year, month) pairs in order from start to end (inclusive)."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1
