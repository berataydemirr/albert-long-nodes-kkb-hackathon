"""
BDDK Weekly Bulletin (BultenHaftalik) collector - Bronze (raw HTML) layer only.

A COMPLETELY DIFFERENT architecture from the Monthly Bulletin: not a
JSON API, but a session-bound, multi-step form flow. See
archive/exploration/recon_bddk_haftalik_findings.md for the
reconnaissance process and the bug that was found (missing
Origin/Referer headers).

Flow, for a single donemId (week):
  1) GET  /bultenhaftalik                          -> cookies + CSRF token
  2) POST /BultenHaftalik/tr/Home/DonemDegistir     -> writes donemId into the session
  3) POST /BultenHaftalik/  (once per tabloId)      -> returns that table's HTML

"donemId" is a single increasing counter (not year+week) - see the
BDDK_WEEKLY_DONEM_START/END comment in config/settings.py.

Usage (from the repo root):
    python -m src.collectors.bddk_weekly --mode test
    python -m src.collectors.bddk_weekly --mode range --start-donem 367 --end-donem 370 --tables 289,290
    python -m src.collectors.bddk_weekly --mode full     # 367..652 - can take HOURS, use with care
"""
import re
import argparse
from typing import Optional

from config.settings import (
    BDDK_WEEKLY_BRONZE_DIR, DEBUG_DIR,
    BDDK_WEEKLY_TABLE_IDS, BDDK_WEEKLY_DONEM_START, BDDK_WEEKLY_DONEM_END,
    TEST_DONEM_ID, TEST_WEEKLY_TABLE_IDS,
)
from src.common.logging_setup import get_logger
from src.common.http import make_session, waf_sleep

logger = get_logger("bddk_weekly")

BASE_URL = "https://www.bddk.org.tr/bultenhaftalik"
INDEX_POST_URL = "https://www.bddk.org.tr/BultenHaftalik/"
DONEM_URL = "https://www.bddk.org.tr/BultenHaftalik/tr/Home/DonemDegistir"

# Origin/Referer are required for the server's anti-forgery check (confirmed
# via reconnaissance - without them the request returns 200 but silently
# changes nothing).
EXTRA_HEADERS = {
    "Origin": "https://www.bddk.org.tr",
    "Referer": BASE_URL,
}

TOKEN_RE = re.compile(r"name=['\"]__RequestVerificationToken['\"][^>]*value=['\"]([^'\"]*)['\"]")


def extract_token(html: str) -> Optional[str]:
    m = TOKEN_RE.search(html)
    return m.group(1) if m else None


def get_bronze_path(donem_id: int, table_id: int):
    d = BDDK_WEEKLY_BRONZE_DIR / str(donem_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"tablo_{table_id}.html"


def is_table_active(html: str, table_id: int) -> bool:
    return f'id="tabloListesiItem-{table_id}"' in html and "activatedItem" in html.split(f'id="tabloListesiItem-{table_id}"')[0][-200:]


def fetch_donem(session, donem_id: int, table_ids: list, stats: dict) -> None:
    missing = [t for t in table_ids if not get_bronze_path(donem_id, t).exists()]
    for t in table_ids:
        if get_bronze_path(donem_id, t).exists():
            stats["bronze_cached"] += 1

    if not missing:
        logger.debug(f"Period {donem_id}: all tables already cached, no network calls made.")
        return

    logger.info(f"Period {donem_id}: establishing session ({len(missing)} tables to fetch)")
    r1 = session.get(BASE_URL, timeout=20)
    token = extract_token(r1.text)
    if not token:
        logger.error(f"Period {donem_id}: could not get CSRF token from the initial page.")
        stats["error"] += 1
        return

    r2 = session.post(
        DONEM_URL,
        data={"donemId": str(donem_id), "__RequestVerificationToken": token},
        headers={**EXTRA_HEADERS, "X-Requested-With": "XMLHttpRequest"},
        timeout=20,
    )
    waf_sleep()
    token = extract_token(r2.text)
    if r2.status_code != 200 or not token:
        logger.error(f"Period {donem_id}: DonemDegistir failed (HTTP {r2.status_code}).")
        stats["error"] += 1
        return

    for table_id in missing:
        r3 = session.post(
            INDEX_POST_URL,
            data={"tabloId": str(table_id), "__RequestVerificationToken": token},
            headers=EXTRA_HEADERS,
            timeout=20,
        )
        waf_sleep()

        if r3.status_code == 200 and is_table_active(r3.text, table_id):
            bronze_path = get_bronze_path(donem_id, table_id)
            with open(bronze_path, "w", encoding="utf-8") as f:
                f.write(r3.text)
            stats["bronze_fetched"] += 1
            logger.info(f"Period {donem_id} Table {table_id}: written to bronze.")
        else:
            debug_file = DEBUG_DIR / f"error_bddk_weekly_donem{donem_id}_tablo{table_id}.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(r3.text)
            stats["error"] += 1
            logger.error(f"Period {donem_id} Table {table_id}: table did not activate, error logged: {debug_file}")

        new_token = extract_token(r3.text)
        if new_token:
            token = new_token


def run(donem_ids, table_ids: list) -> dict:
    session = make_session(extra_headers=EXTRA_HEADERS)
    stats = {"bronze_cached": 0, "bronze_fetched": 0, "error": 0}

    for donem_id in donem_ids:
        fetch_donem(session, donem_id, table_ids, stats)

    logger.info("=" * 50)
    logger.info("BRONZE FETCH SUMMARY (bddk_weekly)")
    logger.info(f"Newly fetched tables : {stats['bronze_fetched']}")
    logger.info(f"Read from cache      : {stats['bronze_cached']}")
    logger.info(f"Errors / failures    : {stats['error']}")
    logger.info("=" * 50)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK Weekly Bulletin bronze collector")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-donem", type=int)
    parser.add_argument("--end-donem", type=int)
    parser.add_argument("--tables", type=str, help="comma-separated table ids, e.g. 289,290")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info(f"TEST MODE: Period {TEST_DONEM_ID} (2021-01-08, week 1), Tables {TEST_WEEKLY_TABLE_IDS}")
        run([TEST_DONEM_ID], TEST_WEEKLY_TABLE_IDS)
    elif args.mode == "full":
        logger.info(f"FULL SCAN MODE: Period {BDDK_WEEKLY_DONEM_START}..{BDDK_WEEKLY_DONEM_END}, Tables {BDDK_WEEKLY_TABLE_IDS}")
        run(range(BDDK_WEEKLY_DONEM_START, BDDK_WEEKLY_DONEM_END + 1), BDDK_WEEKLY_TABLE_IDS)
    else:
        if args.start_donem is None or args.end_donem is None:
            raise SystemExit("--mode range requires --start-donem and --end-donem")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_WEEKLY_TABLE_IDS
        logger.info(f"RANGE MODE: Period {args.start_donem}..{args.end_donem}, Tables: {tables}")
        run(range(args.start_donem, args.end_donem + 1), tables)
