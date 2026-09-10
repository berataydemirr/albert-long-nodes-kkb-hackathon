"""
Checks data freshness (no network): does the gold layer's date coverage
still reach the hackathon's target end date, and are any silver files
suspiciously old (might mean a newer bronze fetch was never re-run
after BDDK/EVDS/TUIK published a new month). Doesn't fetch anything -
purely compares what's already on disk against config/settings.py's
target range and today's date.

Usage (from the repo root):
    python -m src.gold.check_freshness
"""
import argparse
import datetime as dt

import duckdb

from config.settings import GOLD_DIR, SILVER_DIR, FULL_END_YEAR, FULL_END_MONTH
from src.common.logging_setup import get_logger

logger = get_logger("check_freshness")

# A silver file untouched for longer than this is worth a second look -
# not necessarily wrong (a closed historical month never changes), just
# a prompt to check whether a newer period exists upstream now.
STALE_SILVER_DAYS = 30


def check_gold_coverage() -> dict:
    """Confirms gold_fact_long's last Period actually reaches the
    configured FULL_END_YEAR/FULL_END_MONTH - catches the case where a
    rebuild silently ran on stale/partial silver data."""
    db_path = GOLD_DIR / "lakehouse.duckdb"
    expected_last_period = f"{FULL_END_YEAR:04d}-{FULL_END_MONTH:02d}"

    if not db_path.exists():
        logger.error(f"check_gold_coverage: gold database missing at {db_path}")
        return {"ok": False, "reason": "gold database missing", "expected_last_period": expected_last_period, "actual_last_period": None}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT MAX(Period) FROM gold_fact_long").fetchone()
        actual_last = row[0] if row else None
    finally:
        con.close()

    ok = actual_last == expected_last_period
    logger.info(f"check_gold_coverage: expected_last={expected_last_period}, actual_last={actual_last}, ok={ok}")
    if not ok:
        logger.warning(f"check_gold_coverage: gold_fact_long does NOT reach {expected_last_period} (got {actual_last}) - rebuild may be needed")
    return {"ok": ok, "expected_last_period": expected_last_period, "actual_last_period": actual_last}


def check_silver_file_ages() -> list[dict]:
    """Flags silver CSVs whose mtime is older than STALE_SILVER_DAYS."""
    now = dt.datetime.now()
    all_files = list(SILVER_DIR.rglob("*.csv"))
    stale = []
    for csv_path in all_files:
        age_days = (now - dt.datetime.fromtimestamp(csv_path.stat().st_mtime)).days
        if age_days > STALE_SILVER_DAYS:
            stale.append({"path": str(csv_path.relative_to(SILVER_DIR)), "age_days": age_days})
    logger.info(f"check_silver_file_ages: {len(stale)}/{len(all_files)} silver files older than {STALE_SILVER_DAYS} days")
    return stale


def run() -> dict:
    coverage = check_gold_coverage()
    stale_files = check_silver_file_ages()
    result = {"coverage": coverage, "stale_files": stale_files}

    logger.info("=" * 50)
    logger.info("FRESHNESS CHECK SUMMARY")
    logger.info(f"Gold coverage OK       : {coverage['ok']}")
    logger.info(f"Expected last period   : {coverage['expected_last_period']}")
    logger.info(f"Actual last period     : {coverage['actual_last_period']}")
    logger.info(f"Silver files > {STALE_SILVER_DAYS}d old : {len(stale_files)}")
    logger.info("=" * 50)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description="Check gold/silver data freshness (no network)").parse_args()
    run()
