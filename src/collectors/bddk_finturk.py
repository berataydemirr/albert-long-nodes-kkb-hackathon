"""
BDDK FinTurk (by province) collector - Bronze (raw JSON) layer only.

Source: https://www.bddk.org.tr/BultenFinturk/tr/Home/VeriGetir (POST)
The payload shape was read directly from the page's own inline JS
(function VeriGetir()): {tabloNo, donem, tarafList[0]=10001,
sehirList[0]='HEPSİ'}. "HEPSİ" (ALL) returns EVERY province (81
provinces + "abroad" = 82 rows) in a SINGLE request - no need to query
province by province (confirmed live).

Response shape is the SAME jqGrid format as the Monthly Bulletin
({"Json": {"data": {"rows": [...], "total": ...}}}) - src/common/jqgrid.py
is reused here unchanged.

Like the Weekly Bulletin, the server also checks Origin/Referer headers
- added here too to stay on the safe side.

Usage (from the repo root):
    python -m src.collectors.bddk_finturk --mode test
    python -m src.collectors.bddk_finturk --mode range --start-year 2021 --start-q 3 --end-year 2021 --end-q 12 --tables 1,2
    python -m src.collectors.bddk_finturk --mode full     # 2021-3..2026-6, tables 1-7
"""
import json
import argparse
from typing import Optional

from config.settings import (
    BDDK_FINTURK_BRONZE_DIR, DEBUG_DIR,
    BDDK_FINTURK_TABLE_IDS, BDDK_FINTURK_SEHIR_HEPSI, BDDK_TARAF_SEKTOR,
    BDDK_FINTURK_START, BDDK_FINTURK_END,
    TEST_FINTURK_DONEM, TEST_FINTURK_TABLE_IDS,
    quarter_range,
)
from src.common.logging_setup import get_logger
from src.common.http import make_session, waf_sleep
from src.common.jqgrid import get_total_pages

logger = get_logger("bddk_finturk")

URL = "https://www.bddk.org.tr/BultenFinturk/tr/Home/VeriGetir"
EXTRA_HEADERS = {
    "Origin": "https://www.bddk.org.tr",
    "Referer": "https://www.bddk.org.tr/bultenfinturk",
}


def get_bronze_path(donem: str, table_id: int):
    d = BDDK_FINTURK_BRONZE_DIR / donem
    d.mkdir(parents=True, exist_ok=True)
    return d / f"tablo_{table_id}.json"


def fetch_table(session, donem: str, table_id: int) -> Optional[dict]:
    payload = {
        "tabloNo": str(table_id),
        "donem": donem,
        "tarafList[0]": BDDK_TARAF_SEKTOR,
        "sehirList[0]": BDDK_FINTURK_SEHIR_HEPSI,
    }
    try:
        response = session.post(URL, data=payload, headers=EXTRA_HEADERS, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Received HTTP {response.status_code}: Period={donem} Table={table_id}")
            return None
        return response.json()
    except Exception as e:
        logger.error(f"Fetch error Period={donem} Table={table_id}: {e}")
        return None


def run(donems, table_ids: list) -> dict:
    session = make_session(extra_headers=EXTRA_HEADERS)
    stats = {"bronze_cached": 0, "bronze_fetched": 0, "error": 0}

    for donem in donems:
        for table_id in table_ids:
            bronze_path = get_bronze_path(donem, table_id)

            if bronze_path.exists():
                stats["bronze_cached"] += 1
                logger.debug(f"Bronze cache hit: Period={donem} Table={table_id}")
                continue

            logger.info(f"Fetching from API: Period={donem} Table={table_id} (ALL provinces)")
            json_data = fetch_table(session, donem, table_id)
            waf_sleep()

            if json_data and json_data.get("success"):
                total_pages = get_total_pages(json_data)
                if total_pages > 1:
                    logger.warning(
                        f"Period={donem} Table={table_id}: found {total_pages} pages, "
                        f"but this collector assumes a single page (unexpected for an ALL-provinces query) - please check."
                    )
                with open(bronze_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                stats["bronze_fetched"] += 1
            else:
                debug_file = DEBUG_DIR / f"error_bddk_finturk_{donem}_tablo{table_id}.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(json_data if json_data else {"error": "Request failed"}, f, ensure_ascii=False, indent=2)
                stats["error"] += 1
                logger.error(f"Error logged: {debug_file}")

    logger.info("=" * 50)
    logger.info("BRONZE FETCH SUMMARY (bddk_finturk)")
    logger.info(f"Newly fetched tables : {stats['bronze_fetched']}")
    logger.info(f"Read from cache      : {stats['bronze_cached']}")
    logger.info(f"Errors / failures    : {stats['error']}")
    logger.info("=" * 50)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK FinTurk bronze collector")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--start-q", type=int, choices=[3, 6, 9, 12])
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-q", type=int, choices=[3, 6, 9, 12])
    parser.add_argument("--tables", type=str, help="comma-separated table ids, e.g. 1,2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info(f"TEST MODE: Period {TEST_FINTURK_DONEM}, Tables {TEST_FINTURK_TABLE_IDS}")
        run([TEST_FINTURK_DONEM], TEST_FINTURK_TABLE_IDS)
    elif args.mode == "full":
        donems = list(quarter_range(*BDDK_FINTURK_START, *BDDK_FINTURK_END))
        logger.info(f"FULL SCAN MODE: {donems[0]}..{donems[-1]} ({len(donems)} periods), Tables 1-7")
        run(donems, BDDK_FINTURK_TABLE_IDS)
    else:
        missing = [n for n in ("start_year", "start_q", "end_year", "end_q") if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"--mode range requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_FINTURK_TABLE_IDS
        donems = list(quarter_range(args.start_year, args.start_q, args.end_year, args.end_q))
        logger.info(f"RANGE MODE: {donems[0]}..{donems[-1]}, Tables: {tables}")
        run(donems, tables)
