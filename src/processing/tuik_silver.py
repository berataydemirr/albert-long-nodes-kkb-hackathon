"""
TUIK bronze (SDMX-ML XML) -> silver conversion. Network-free.

Usage (from the repo root):
    python -m src.processing.tuik_silver
"""
import re
import argparse
import pandas as pd

from config.settings import TUIK_BRONZE_DIR, TUIK_SILVER_DIR, FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH
from config.tuik_series import SERIES
from src.common.logging_setup import get_logger, log_stats_summary

logger = get_logger("tuik_silver")

OBS_RE = re.compile(
    r'<generic:ObsDimension id="TIME_PERIOD" value="([^"]+)"\s*/>'
    r'<generic:ObsValue value="([^"]+)"\s*/>'
)


def get_bronze_path(series_code: str):
    return TUIK_BRONZE_DIR / f"{series_code}.xml"


def get_silver_path(series_code: str):
    TUIK_SILVER_DIR.mkdir(parents=True, exist_ok=True)
    return TUIK_SILVER_DIR / f"{series_code}.csv"


def _period_in_range(period: str) -> bool:
    """True if `period` ('YYYY' or 'YYYY-MM') falls within
    FULL_START_YEAR/MONTH .. FULL_END_YEAR/MONTH. Works for both
    granularities instead of assuming every TUIK series is monthly."""
    year = int(period[:4])
    if year < FULL_START_YEAR or year > FULL_END_YEAR:
        return False
    has_month = len(period) > 4
    if year == FULL_START_YEAR and has_month and int(period[5:7]) < FULL_START_MONTH:
        return False
    if year == FULL_END_YEAR and has_month and int(period[5:7]) > FULL_END_MONTH:
        return False
    return True


def parse_series(series_code: str) -> bool:
    bronze_path = get_bronze_path(series_code)
    if not bronze_path.exists():
        return False

    with open(bronze_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    obs = OBS_RE.findall(xml_text)
    if not obs:
        return False

    df = pd.DataFrame(obs, columns=["Period", "Value"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.drop_duplicates(subset="Period").sort_values("Period").reset_index(drop=True)

    # The TUIK API interprets startPeriod/endPeriod loosely (can return one
    # extra period) - trim any rows that fall outside the requested
    # 2021-01..2026-06.
    #
    # BUG FOUND WHILE ADDING AN ANNUAL SERIES: this used to compare Period
    # against "2021-01"/"2026-06" as plain strings. That's correct for a
    # monthly series ("2021-01" >= "2021-01"), but TUIK_LIFE_EXPECTANCY's
    # periods are bare years ("2021", "2022", ...) - and in Python string
    # comparison "2021" < "2021-01" (a shorter string that's a prefix of a
    # longer one always sorts first), so 2021's real, valid observation was
    # silently trimmed away as "before the start date". Fixed to parse the
    # year (and month, only when present) instead of comparing raw strings.
    df = df[df["Period"].apply(_period_in_range)].reset_index(drop=True)

    df.to_csv(get_silver_path(series_code), index=False, encoding="utf-8")
    return True


def run() -> dict:
    stats = {"parsed": 0, "empty_or_missing": 0}
    for series in SERIES:
        code = series["code"]
        if parse_series(code):
            logger.info(f"Written to silver: {code}")
            stats["parsed"] += 1
        else:
            logger.warning(f"Series empty or bronze missing: {code}")
            stats["empty_or_missing"] += 1

    log_stats_summary(logger, "tuik", stats)
    return stats


if __name__ == "__main__":
    argparse.ArgumentParser(description="TUIK bronze->silver (no network)").parse_args()
    run()
