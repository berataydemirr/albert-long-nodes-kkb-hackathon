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
from src.common.logging_setup import get_logger

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
    # extra month) - trim any rows that fall outside the requested 2021-01..2026-06.
    start_key = f"{FULL_START_YEAR}-{FULL_START_MONTH:02d}"
    end_key = f"{FULL_END_YEAR}-{FULL_END_MONTH:02d}"
    df = df[(df["Period"] >= start_key) & (df["Period"] <= end_key)].reset_index(drop=True)

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

    logger.info("=" * 50)
    logger.info("SILVER PARSE SUMMARY (tuik)")
    logger.info(f"Newly parsed  : {stats['parsed']}")
    logger.info(f"Empty/missing : {stats['empty_or_missing']}")
    logger.info("=" * 50)
    return stats


if __name__ == "__main__":
    argparse.ArgumentParser(description="TUIK bronze->silver (no network)").parse_args()
    run()
