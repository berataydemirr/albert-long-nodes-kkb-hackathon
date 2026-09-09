"""
BDDK FinTurk bronze -> silver conversion.

Network-free: only reads bronze JSON files from disk. Since the response
shape matches the Monthly Bulletin, src/common/jqgrid.py is reused
unchanged. Because an "ALL provinces" query returns every province on a
single page (confirmed live: 82 rows/page), this module reads the one
bronze file directly instead of looping through pagination like the
monthly module does.

Usage (from the repo root):
    python -m src.processing.bddk_finturk_silver --mode test
    python -m src.processing.bddk_finturk_silver --mode range --start-year 2021 --start-q 3 --end-year 2021 --end-q 12 --tables 1,2
    python -m src.processing.bddk_finturk_silver --mode full
"""
import json
import argparse
import pandas as pd

from config.settings import (
    BDDK_FINTURK_BRONZE_DIR, BDDK_FINTURK_SILVER_DIR, BDDK_FINTURK_TABLE_IDS,
    BDDK_FINTURK_START, BDDK_FINTURK_END,
    TEST_FINTURK_DONEM, TEST_FINTURK_TABLE_IDS,
    quarter_range,
)
from src.common.logging_setup import get_logger
from src.common.jqgrid import get_columns_and_rows, rows_to_records

logger = get_logger("bddk_finturk_silver")


def get_bronze_path(donem: str, table_id: int):
    return BDDK_FINTURK_BRONZE_DIR / donem / f"tablo_{table_id}.json"


def get_silver_path(donem: str, table_id: int):
    d = BDDK_FINTURK_SILVER_DIR / f"tablo_{table_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{donem}.csv"


def parse_table(donem: str, table_id: int) -> bool:
    bronze_path = get_bronze_path(donem, table_id)
    if not bronze_path.exists():
        return False

    with open(bronze_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    col_names, rows = get_columns_and_rows(json_data)
    if not rows or not col_names:
        return False

    records = rows_to_records(col_names, rows)
    df = pd.DataFrame(records)
    df.to_csv(get_silver_path(donem, table_id), index=False, encoding="utf-8")
    return True


def run(donems, table_ids: list) -> dict:
    stats = {"parsed": 0, "empty_or_missing": 0, "skipped_exists": 0}

    for donem in donems:
        for table_id in table_ids:
            silver_path = get_silver_path(donem, table_id)
            if silver_path.exists():
                stats["skipped_exists"] += 1
                continue

            if parse_table(donem, table_id):
                logger.info(f"Written to silver: T{table_id} Period={donem}")
                stats["parsed"] += 1
            else:
                logger.warning(f"Table empty or bronze missing: Period={donem} T{table_id}")
                stats["empty_or_missing"] += 1

    logger.info("=" * 50)
    logger.info("SILVER PARSE SUMMARY (bddk_finturk)")
    logger.info(f"Newly parsed      : {stats['parsed']}")
    logger.info(f"Already existed   : {stats['skipped_exists']}")
    logger.info(f"Empty/missing     : {stats['empty_or_missing']}")
    logger.info("=" * 50)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK FinTurk bronze->silver (no network)")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--start-q", type=int, choices=[3, 6, 9, 12])
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-q", type=int, choices=[3, 6, 9, 12])
    parser.add_argument("--tables", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info(f"TEST MODE: Period {TEST_FINTURK_DONEM}, Tables {TEST_FINTURK_TABLE_IDS}")
        run([TEST_FINTURK_DONEM], TEST_FINTURK_TABLE_IDS)
    elif args.mode == "full":
        run(list(quarter_range(*BDDK_FINTURK_START, *BDDK_FINTURK_END)), BDDK_FINTURK_TABLE_IDS)
    else:
        missing = [n for n in ("start_year", "start_q", "end_year", "end_q") if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"--mode range requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_FINTURK_TABLE_IDS
        run(list(quarter_range(args.start_year, args.start_q, args.end_year, args.end_q)), tables)
