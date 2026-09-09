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

# --- Data layers (medallion architecture: bronze=raw, silver=clean, gold=analysis-ready) ---
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DEBUG_DIR = DATA_DIR / "debug"
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
