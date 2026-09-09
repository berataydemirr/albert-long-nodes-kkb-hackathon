"""
BDDK Monthly Bulletin (BultenAylik) collector - Bronze (raw data) layer only.

Source: https://www.bddk.org.tr/BultenAylik/tr/Home/BasitRaporGetir
The payload shape was found via archive/exploration/test_bddk3.py: flat
field names (yil, ay, tabloNo, taraf, paraBirimi) + an X-Requested-With
header. See archive/exploration/README.md.

This module ONLY handles the network/fetch step (writes to bronze).
Parsing bronze into silver lives in src/processing/bddk_monthly_silver.py
- kept separate on purpose, since that step should be network-free and
safe to re-run as many times as needed.

Usage (from the repo root):
    python -m src.collectors.bddk_monthly --mode test
    python -m src.collectors.bddk_monthly --mode range --start-year 2021 --start-month 1 --end-year 2021 --end-month 2 --tables 1,2,3
    python -m src.collectors.bddk_monthly --mode full     # 2021-01..2026-06, tables 1-17 - can take HOURS, use with care
"""
import json
import argparse
from typing import Optional

from config.settings import (
    BDDK_MONTHLY_BRONZE_DIR, DEBUG_DIR,
    BDDK_TABLE_IDS, BDDK_TARAF_SEKTOR, BDDK_PARA_BIRIMI_DEFAULT,
    FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH,
    TEST_YEAR, TEST_MONTH, TEST_TABLE_IDS,
    month_range,
)
from src.common.logging_setup import get_logger
from src.common.http import make_session, waf_sleep
from src.common.jqgrid import get_total_pages

logger = get_logger("bddk_monthly")

URL = "https://www.bddk.org.tr/BultenAylik/tr/Home/BasitRaporGetir"


def get_bronze_path(year: int, month: int, table_id: int, page: int = 1):
    ym_dir = BDDK_MONTHLY_BRONZE_DIR / str(year) / f"{month:02d}"
    ym_dir.mkdir(parents=True, exist_ok=True)
    return ym_dir / f"tablo_{table_id}_p{page}.json"


def fetch_page(session, year: int, month: int, table_id: int, page: int) -> Optional[dict]:
    payload = {
        "yil": str(year),
        "ay": str(month),
        "tabloNo": str(table_id),
        "paraBirimi": BDDK_PARA_BIRIMI_DEFAULT,
        "taraf": BDDK_TARAF_SEKTOR,
        "page": str(page),
        "rows": "1000",  # high row count to minimize pagination
    }
    try:
        response = session.post(URL, data=payload, timeout=20)
        if response.status_code != 200:
            logger.warning(f"Received HTTP {response.status_code}: Y{year}-M{month}-T{table_id} Page {page}")
            return None
        return response.json()
    except Exception as e:
        logger.error(f"Fetch error Y{year}-M{month}-T{table_id} Page {page}: {e}")
        return None


def fetch_table(session, year: int, month: int, table_id: int, stats: dict) -> bool:
    """Fetches all pages of a table (or reads them from cache). Returns False on failure."""
    page = 1
    total_pages = 1

    while page <= total_pages:
        bronze_path = get_bronze_path(year, month, table_id, page)

        if bronze_path.exists():
            with open(bronze_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            total_pages = get_total_pages(cached_data)
            stats["bronze_cached"] += 1
            logger.debug(f"Bronze cache hit: Y{year}-M{month}-T{table_id} Page {page}")
        else:
            logger.info(f"Fetching from API: Y{year}-M{month}-T{table_id} Page {page}")
            json_data = fetch_page(session, year, month, table_id, page)
            waf_sleep()

            if json_data and json_data.get("success"):
                with open(bronze_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                stats["bronze_fetched"] += 1
                total_pages = get_total_pages(json_data)
            else:
                debug_file = DEBUG_DIR / f"error_bddk_monthly_Y{year}_M{month}_T{table_id}_p{page}.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(json_data if json_data else {"error": "Request failed"}, f, ensure_ascii=False, indent=2)
                stats["error"] += 1
                logger.error(f"Error logged: {debug_file}")
                return False

        page += 1

    return True


def run(years_months, table_ids: list) -> dict:
    session = make_session()
    stats = {"bronze_cached": 0, "bronze_fetched": 0, "error": 0}

    for year, month in years_months:
        for table_id in table_ids:
            ok = fetch_table(session, year, month, table_id, stats)
            if not ok:
                logger.warning(f"Could not fetch table, skipping: Y{year}-M{month}-T{table_id}")

    logger.info("=" * 50)
    logger.info("BRONZE FETCH SUMMARY (bddk_monthly)")
    logger.info(f"Newly fetched pages : {stats['bronze_fetched']}")
    logger.info(f"Read from cache     : {stats['bronze_cached']}")
    logger.info(f"Errors / failures   : {stats['error']}")
    logger.info("=" * 50)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK Monthly Bulletin bronze collector")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--start-month", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-month", type=int)
    parser.add_argument("--tables", type=str, help="comma-separated table ids, e.g. 1,2,5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info("TEST MODE: 2021-01, Tables 1-3")
        run([(TEST_YEAR, TEST_MONTH)], TEST_TABLE_IDS)
    elif args.mode == "full":
        logger.info(f"FULL SCAN MODE: {FULL_START_YEAR}-{FULL_START_MONTH:02d} .. {FULL_END_YEAR}-{FULL_END_MONTH:02d}, Tables 1-17")
        run(month_range(FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH), BDDK_TABLE_IDS)
    else:
        missing = [n for n in ("start_year", "start_month", "end_year", "end_month") if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"--mode range requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_TABLE_IDS
        logger.info(f"RANGE MODE: {args.start_year}-{args.start_month:02d} .. {args.end_year}-{args.end_month:02d}, Tables: {tables}")
        run(month_range(args.start_year, args.start_month, args.end_year, args.end_month), tables)
