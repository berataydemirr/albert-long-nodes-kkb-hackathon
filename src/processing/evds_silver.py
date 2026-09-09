"""
EVDS bronze -> silver conversion. Network-free.

WHY THIS SCRIPT EXISTS: unlike BDDK (messy jqGrid JSON) or TUIK (SDMX
XML), the `evds` library already returns a tidy DataFrame, so the
collector wrote it straight to bronze with no separate cleaning step.
That looked "clean enough" but hid a real bug: EVDS's own date format
is inconsistent by frequency and is NOT zero-padded for monthly series
('2021-1', not '2021-01'). BDDK silver uses 'YYYY_MM' and TUIK silver
uses 'YYYY-MM' - three different date shapes across sources would have
silently broken every join in the gold layer. This script normalizes
EVDS to the same 'YYYY-MM' (monthly) / 'YYYY-MM-DD' (daily) convention
used everywhere else, detecting frequency per-series instead of
assuming one.

Usage (from the repo root):
    python -m src.processing.evds_silver
"""
import argparse
import re

import pandas as pd

from config.settings import (
    EVDS_BRONZE_DIR, EVDS_SILVER_DIR,
    FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH,
)
from config.evds_series import SERIES
from src.common.logging_setup import get_logger

logger = get_logger("evds_silver")

MONTHLY_RE = re.compile(r"^\d{4}-\d{1,2}$")   # e.g. '2021-1'
DAILY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")  # e.g. '04-01-2021'


def get_bronze_path(series_code: str):
    return EVDS_BRONZE_DIR / f"{series_code.replace('.', '_')}.csv"


def get_silver_path(series_code: str):
    EVDS_SILVER_DIR.mkdir(parents=True, exist_ok=True)
    return EVDS_SILVER_DIR / f"{series_code.replace('.', '_')}.csv"


def _detect_frequency(sample_dates: pd.Series) -> str:
    """Looks at the first few raw date strings to tell monthly ('2021-1')
    from daily ('04-01-2021') series - EVDS mixes both across the
    registry depending on how the source data is published."""
    head = sample_dates.dropna().astype(str).head(5)
    if head.apply(lambda s: bool(MONTHLY_RE.match(s))).all():
        return "monthly"
    if head.apply(lambda s: bool(DAILY_RE.match(s))).all():
        return "daily"
    raise ValueError(f"Unrecognized EVDS date format, first values: {head.tolist()}")


def parse_series(series_code: str) -> bool:
    bronze_path = get_bronze_path(series_code)
    if not bronze_path.exists():
        return False

    raw = pd.read_csv(bronze_path, encoding="utf-8")
    date_col, value_col = raw.columns[0], raw.columns[1]  # 'Tarih', '<SERIES_CODE>'

    freq = _detect_frequency(raw[date_col])

    df = pd.DataFrame()
    if freq == "monthly":
        # '2021-1' -> ('2021','1') -> '2021-01'
        parts = raw[date_col].astype(str).str.split("-", n=1, expand=True)
        df["Period"] = parts[0] + "-" + parts[1].str.zfill(2)
    else:  # daily
        df["Period"] = pd.to_datetime(raw[date_col], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")

    df["Value"] = pd.to_numeric(raw[value_col], errors="coerce")
    df["Series_Code"] = series_code
    df = df.dropna(subset=["Value"]).drop_duplicates(subset="Period").sort_values("Period").reset_index(drop=True)

    # Trim to the hackathon's date range - same guard tuik_silver.py uses,
    # needed because monthly bounds don't apply cleanly to daily rows otherwise.
    start_key = f"{FULL_START_YEAR}-{FULL_START_MONTH:02d}"
    end_key = f"{FULL_END_YEAR}-{FULL_END_MONTH:02d}-31"
    df = df[(df["Period"] >= start_key) & (df["Period"] <= end_key)].reset_index(drop=True)

    df.to_csv(get_silver_path(series_code), index=False, encoding="utf-8")
    logger.info(f"Written to silver ({freq}): {series_code} ({len(df)} rows)")
    return True


def run() -> dict:
    stats = {"parsed": 0, "empty_or_missing": 0}
    for series in SERIES:
        code = series["code"]
        if parse_series(code):
            stats["parsed"] += 1
        else:
            logger.warning(f"Bronze missing, skipped: {code}")
            stats["empty_or_missing"] += 1

    logger.info("=" * 50)
    logger.info("SILVER PARSE SUMMARY (evds)")
    logger.info(f"Newly parsed  : {stats['parsed']}")
    logger.info(f"Empty/missing : {stats['empty_or_missing']}")
    logger.info("=" * 50)
    return stats


if __name__ == "__main__":
    argparse.ArgumentParser(description="EVDS bronze->silver (no network)").parse_args()
    run()
