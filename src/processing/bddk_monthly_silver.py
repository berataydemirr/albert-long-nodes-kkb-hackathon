"""
BDDK Monthly Bulletin bronze -> silver conversion.

Deliberately NETWORK-FREE: only reads bronze JSON files from disk, makes
no HTTP requests. This separation is what let us verify the actual
parsing bug (see src/common/jqgrid.py) in seconds, without touching the
network at all.

Usage (from the repo root):
    python -m src.processing.bddk_monthly_silver --mode test
    python -m src.processing.bddk_monthly_silver --mode range --start-year 2021 --start-month 1 --end-year 2021 --end-month 2 --tables 1,2,3
    python -m src.processing.bddk_monthly_silver --mode full
"""
import json
import argparse
import pandas as pd

from config.settings import (
    BDDK_MONTHLY_BRONZE_DIR, BDDK_MONTHLY_SILVER_DIR, BDDK_TABLE_IDS,
    FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH,
    TEST_YEAR, TEST_MONTH, TEST_TABLE_IDS,
    month_range,
)
from src.common.logging_setup import get_logger, log_stats_summary
from src.common.jqgrid import get_columns_and_rows, get_total_pages, rows_to_records

logger = get_logger("bddk_monthly_silver")


def get_bronze_path(year: int, month: int, table_id: int, page: int = 1):
    return BDDK_MONTHLY_BRONZE_DIR / str(year) / f"{month:02d}" / f"tablo_{table_id}_p{page}.json"


def get_silver_path(year: int, month: int, table_id: int):
    d = BDDK_MONTHLY_SILVER_DIR / f"tablo_{table_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{year}_{month:02d}.csv"


def parse_table(year: int, month: int, table_id: int) -> bool:
    """Reads all bronze pages and merges them into a single silver CSV. Returns False on failure."""
    page = 1
    all_rows = []
    col_names: list = []

    while True:
        bronze_path = get_bronze_path(year, month, table_id, page)
        if not bronze_path.exists():
            break

        with open(bronze_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        cols, rows = get_columns_and_rows(json_data)
        if page == 1:
            col_names = cols
        all_rows.extend(rows)

        total_pages = get_total_pages(json_data)
        if page >= total_pages:
            break
        page += 1

    if not all_rows or not col_names:
        return False

    records = rows_to_records(col_names, all_rows)
    df = pd.DataFrame(records)
    df.to_csv(get_silver_path(year, month, table_id), index=False, encoding="utf-8")
    return True


def run(years_months, table_ids: list) -> dict:
    stats = {"parsed": 0, "empty_or_missing": 0, "skipped_exists": 0}

    for year, month in years_months:
        for table_id in table_ids:
            silver_path = get_silver_path(year, month, table_id)
            if silver_path.exists():
                stats["skipped_exists"] += 1
                continue

            if parse_table(year, month, table_id):
                logger.info(f"Written to silver: T{table_id} {year}-{month:02d}")
                stats["parsed"] += 1
            else:
                logger.warning(f"Table empty or bronze missing: Y{year}-M{month}-T{table_id}")
                stats["empty_or_missing"] += 1

    log_stats_summary(logger, "bddk_monthly", stats)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK Monthly Bulletin bronze->silver (no network)")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--start-month", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-month", type=int)
    parser.add_argument("--tables", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info("TEST MODE: 2021-01, Tables 1-3")
        run([(TEST_YEAR, TEST_MONTH)], TEST_TABLE_IDS)
    elif args.mode == "full":
        run(month_range(FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH), BDDK_TABLE_IDS)
    else:
        missing = [n for n in ("start_year", "start_month", "end_year", "end_month") if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"--mode range requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_TABLE_IDS
        run(month_range(args.start_year, args.start_month, args.end_year, args.end_month), tables)
