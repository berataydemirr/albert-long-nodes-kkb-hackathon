"""
TCMB EVDS (Central Bank Electronic Data Distribution System) collector.

STATUS: config/evds_series.py defines 6 real series (house price index,
housing loan interest rate, CBRT policy rate, CPI, USD rate, industrial
production) - chosen to match the hackathon brief's causality example
(interest rates <-> house prices), found via --discover and confirmed
with get_series().

--discover mode uses the evds library's get_sub_categories() method
(directly verified: called with no parameters, it returns ALL data
groups as a flat list of CATEGORY_ID / DATAGROUP_CODE / DATAGROUP_NAME)
to dump real candidate series to data/debug/evds_categories.json. To add
a new series: find its DATAGROUP_CODE, call get_series(datagroup_code)
to confirm the real SERIE_CODE, then add it to config/evds_series.py.

Usage:
    python -m src.collectors.evds                           # all series in config/evds_series.py, full date range
    python -m src.collectors.evds --discover                 # list all data groups
    python -m src.collectors.evds --discover --search konut  # filter by name (e.g. "konut" = housing)
"""
import json
import argparse
from datetime import date

import evds
import pandas as pd

from config.settings import (
    EVDS_API_KEY, EVDS_BRONZE_DIR, DEBUG_DIR,
    FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH,
)
from config.evds_series import SERIES
from src.common.logging_setup import get_logger

logger = get_logger("evds")

# BUGFIX (found via a live run): the TCMB EVDS server caps daily-frequency
# series at ~1000 rows per request - if the requested date range is
# longer than that, it does NOT return an error, it SILENTLY TRUNCATES
# to the last ~1000 rows instead of starting from the requested start
# date. The USD rate series started at 2023-10-05 even though 2021-01-01
# was requested. Fix: fetch each series in separate one-year chunks and
# concatenate them (a one-year chunk at daily frequency is ~365 rows,
# safely under the limit).


def _get_client() -> "evds.evdsAPI":
    if not EVDS_API_KEY:
        raise ValueError("EVDS_API_KEY is not set in the environment (check your .env file).")
    return evds.evdsAPI(EVDS_API_KEY)


def _year_chunks(start_date: str, end_date: str):
    """Splits a 'dd-mm-yyyy' start/end range into one-year (dd-mm-yyyy, dd-mm-yyyy) chunks."""
    sd, sm, sy = (int(x) for x in start_date.split("-"))
    ed, em, ey = (int(x) for x in end_date.split("-"))
    start = date(sy, sm, sd)
    end = date(ey, em, ed)

    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(date(chunk_start.year, 12, 31), end)
        yield chunk_start.strftime("%d-%m-%Y"), chunk_end.strftime("%d-%m-%Y")
        chunk_start = date(chunk_start.year + 1, 1, 1)


def collect_series(series_code: str, start_date: str, end_date: str, force: bool = False) -> bool:
    """Fetches a single EVDS series (in one-year chunks, to stay under the
    server's 1000-row limit) and saves it to data/bronze/evds/{series_code}.csv.
    Skips the fetch if the file already exists (force=False) - same
    checkpoint idea as the BDDK collectors."""
    output_path = EVDS_BRONZE_DIR / f"{series_code.replace('.', '_')}.csv"

    if output_path.exists() and not force:
        logger.debug(f"Bronze cache hit, skipping: {series_code} -> {output_path}")
        return True

    client = _get_client()
    logger.info(f"Fetching from EVDS: '{series_code}' ({start_date} - {end_date}, in yearly chunks)")

    parts = []
    for chunk_start, chunk_end in _year_chunks(start_date, end_date):
        try:
            df_chunk = client.get_data([series_code], startdate=chunk_start, enddate=chunk_end)
        except Exception as e:
            logger.error(f"Fetch error '{series_code}' [{chunk_start}-{chunk_end}]: {e}")
            return False
        if df_chunk is not None and not df_chunk.empty:
            parts.append(df_chunk)
        logger.debug(f"  {series_code} [{chunk_start}-{chunk_end}]: {len(df_chunk) if df_chunk is not None else 0} rows")

    if not parts:
        logger.warning(f"No data returned: {series_code}")
        return False

    df = pd.concat(parts, ignore_index=True).drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(df)} rows)")
    return True


def discover(search: str = "") -> None:
    """Discovers the real EVDS data-group list (instead of guessing series codes) and dumps it to a file."""
    client = _get_client()
    logger.info("Fetching EVDS data groups (get_sub_categories)...")
    df = client.get_sub_categories()

    if search:
        mask = df["DATAGROUP_NAME"].str.contains(search, case=False, na=False)
        df = df[mask]
        logger.info(f"Found {len(df)} matches for '{search}'.")

    out_path = DEBUG_DIR / "evds_categories.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {out_path} ({len(df)} records)")
    logger.info("Review this file to pick the relevant DATAGROUP_CODEs and add them to config/evds_series.py.")


def parse_args():
    parser = argparse.ArgumentParser(description="TCMB EVDS collector")
    parser.add_argument("--discover", action="store_true", help="Discover the data-group list instead of guessing series codes")
    parser.add_argument("--search", type=str, default="", help="With --discover: filter by keyword in the name")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if bronze cache exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.discover:
        discover(search=args.search)
    elif not SERIES:
        raise SystemExit(
            "config/evds_series.py is empty - first search for series with "
            "'--discover --search <keyword>' and add them to the list (the code never guesses series codes)."
        )
    else:
        start_date = f"01-{FULL_START_MONTH:02d}-{FULL_START_YEAR}"
        end_date = f"30-{FULL_END_MONTH:02d}-{FULL_END_YEAR}"
        logger.info(f"{len(SERIES)} series to fetch: {[s['code'] for s in SERIES]}")

        results = {"success": 0, "failed": 0}
        for series in SERIES:
            ok = collect_series(series["code"], start_date, end_date, force=args.force)
            results["success" if ok else "failed"] += 1

        logger.info("=" * 50)
        logger.info("EVDS FETCH SUMMARY")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Failed : {results['failed']}")
        logger.info("=" * 50)
