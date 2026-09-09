"""
BDDK Weekly Bulletin bronze -> silver conversion.

Network-free: only reads bronze HTML files from disk (same fetch/parse
separation principle as the monthly processing module).

Bronze HTML structure (confirmed via reconnaissance, see
archive/exploration/recon_bddk_haftalik_findings.md):
    <table id="Tablo">
      <thead><tr><th></th><th>{table name}</th><th>TP</th><th>YP</th><th>TOPLAM</th></tr></thead>
      <tr class="satir">
        <td>{sequence}</td><td>{item name}</td><td>{TP}</td><td>{YP}</td><td>{TOPLAM}</td>
      </tr>
      ...
    </table>
Numbers arrive in Turkish format (dot as thousands separator, e.g.
"18.805" = 18805).

Usage (from the repo root):
    python -m src.processing.bddk_weekly_silver --mode test
    python -m src.processing.bddk_weekly_silver --mode range --start-donem 367 --end-donem 370 --tables 289,290
    python -m src.processing.bddk_weekly_silver --mode full
"""
import argparse
import pandas as pd
from bs4 import BeautifulSoup

from config.settings import (
    BDDK_WEEKLY_BRONZE_DIR, BDDK_WEEKLY_SILVER_DIR, BDDK_WEEKLY_TABLE_IDS,
    BDDK_WEEKLY_DONEM_START, BDDK_WEEKLY_DONEM_END,
    TEST_DONEM_ID, TEST_WEEKLY_TABLE_IDS,
)
from src.common.logging_setup import get_logger

logger = get_logger("bddk_weekly_silver")


def get_bronze_path(donem_id: int, table_id: int):
    return BDDK_WEEKLY_BRONZE_DIR / str(donem_id) / f"tablo_{table_id}.html"


def get_silver_path(donem_id: int, table_id: int):
    d = BDDK_WEEKLY_SILVER_DIR / f"tablo_{table_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"donem_{donem_id}.csv"


def parse_turkish_number(text: str):
    """'18.805' -> 18805, '1.234,5' -> 1234.5. Returns None for empty/meaningless values."""
    text = text.strip().replace(".", "").replace(",", ".")
    if text in ("", "-"):
        return None
    try:
        val = float(text)
        return int(val) if val.is_integer() else val
    except ValueError:
        return None


def parse_table(donem_id: int, table_id: int) -> bool:
    bronze_path = get_bronze_path(donem_id, table_id)
    if not bronze_path.exists():
        return False

    with open(bronze_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="Tablo")
    if table is None or table.find("thead") is None:
        return False

    header_cells = table.find("thead").find_all("th")
    if len(header_cells) < 3:
        return False
    value_cols = [th.get_text(strip=True) for th in header_cells[2:]]  # ["TP", "YP", "TOPLAM"]

    records = []
    for tr in table.find_all("tr", class_="satir"):
        tds = tr.find_all("td")
        if len(tds) < 2 + len(value_cols):
            continue
        record = {
            "Sequence": tds[0].get_text(strip=True),
            "Item": tds[1].get_text(" ", strip=True),
        }
        for i, col_name in enumerate(value_cols):
            record[col_name] = parse_turkish_number(tds[2 + i].get_text(strip=True))
        records.append(record)

    if not records:
        return False

    df = pd.DataFrame(records)
    df.to_csv(get_silver_path(donem_id, table_id), index=False, encoding="utf-8")
    return True


def run(donem_ids, table_ids: list) -> dict:
    stats = {"parsed": 0, "empty_or_missing": 0, "skipped_exists": 0}

    for donem_id in donem_ids:
        for table_id in table_ids:
            silver_path = get_silver_path(donem_id, table_id)
            if silver_path.exists():
                stats["skipped_exists"] += 1
                continue

            if parse_table(donem_id, table_id):
                logger.info(f"Written to silver: T{table_id} Period{donem_id}")
                stats["parsed"] += 1
            else:
                logger.warning(f"Table empty or bronze missing: Period{donem_id}-T{table_id}")
                stats["empty_or_missing"] += 1

    logger.info("=" * 50)
    logger.info("SILVER PARSE SUMMARY (bddk_weekly)")
    logger.info(f"Newly parsed      : {stats['parsed']}")
    logger.info(f"Already existed   : {stats['skipped_exists']}")
    logger.info(f"Empty/missing     : {stats['empty_or_missing']}")
    logger.info("=" * 50)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="BDDK Weekly Bulletin bronze->silver (no network)")
    parser.add_argument("--mode", choices=["test", "full", "range"], default="test")
    parser.add_argument("--start-donem", type=int)
    parser.add_argument("--end-donem", type=int)
    parser.add_argument("--tables", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        logger.info(f"TEST MODE: Period {TEST_DONEM_ID}, Tables {TEST_WEEKLY_TABLE_IDS}")
        run([TEST_DONEM_ID], TEST_WEEKLY_TABLE_IDS)
    elif args.mode == "full":
        run(range(BDDK_WEEKLY_DONEM_START, BDDK_WEEKLY_DONEM_END + 1), BDDK_WEEKLY_TABLE_IDS)
    else:
        if args.start_donem is None or args.end_donem is None:
            raise SystemExit("--mode range requires --start-donem and --end-donem")
        tables = [int(t) for t in args.tables.split(",")] if args.tables else BDDK_WEEKLY_TABLE_IDS
        run(range(args.start_donem, args.end_donem + 1), tables)
