"""
TUIK (Turkish Statistical Institute) SDMX API collector - Bronze (raw XML) layer only.

Authentication (taken directly from the user's own SDMX Web Service
documentation, not guessed):
  1) POST https://giris.tuik.gov.tr/realms/web/protocol/openid-connect/token
     data: grant_type=password, client_id=nsi-ws-consumer, api_key=<TUIK_API_KEY>
     -> returns access_token in JSON. THE TOKEN EXPIRES IN 300 SECONDS
        (5 min) - a fresh token is fetched on every fetch_series() call,
        never cached.
  2) Data query: GET https://nsiws.tuik.gov.tr/rest/data/TR,{dataflow_id},{version}/{key}
     header: Authorization: Bearer {access_token}
     -> returns SDMX-ML 2.1 Generic Data (XML).

Series keys are REGISTERED in config/tuik_series.py - each one was
resolved via the dataflow -> DSD -> codelist chain and confirmed, not
guessed. Adding a new series means repeating that same process (see README).

Usage (from the repo root):
    python -m src.collectors.tuik
    python -m src.collectors.tuik --force
"""
import re
import argparse

import requests

from config.settings import TUIK_API_KEY, TUIK_BRONZE_DIR, DEBUG_DIR, FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH
from config.tuik_series import SERIES
from src.common.logging_setup import get_logger

logger = get_logger("tuik")

TOKEN_URL = "https://giris.tuik.gov.tr/realms/web/protocol/openid-connect/token"
DATA_BASE_URL = "https://nsiws.tuik.gov.tr/rest/data/TR"


def get_access_token() -> str:
    if not TUIK_API_KEY:
        raise ValueError("TUIK_API_KEY is not set in the environment (check your .env file).")
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "password", "client_id": "nsi-ws-consumer", "api_key": TUIK_API_KEY},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_bronze_path(series_code: str):
    return TUIK_BRONZE_DIR / f"{series_code}.xml"


def fetch_series(dataflow_id: str, version: str, key: str, series_code: str, force: bool = False) -> bool:
    bronze_path = get_bronze_path(series_code)
    if bronze_path.exists() and not force:
        logger.debug(f"Bronze cache hit, skipping: {series_code}")
        return True

    token = get_access_token()  # fresh token every time - expires after 5 minutes
    start = f"{FULL_START_YEAR}-{FULL_START_MONTH:02d}"
    end = f"{FULL_END_YEAR}-{FULL_END_MONTH:02d}"

    url = f"{DATA_BASE_URL},{dataflow_id},{version}/{key}"
    logger.info(f"Fetching from TUIK: {series_code} ({dataflow_id}/{key}) [{start}..{end}]")
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={"startPeriod": start, "endPeriod": end},
            timeout=30,
        )
    except Exception as e:
        logger.error(f"Fetch error '{series_code}': {e}")
        return False

    if resp.status_code != 200:
        debug_file = DEBUG_DIR / f"error_tuik_{series_code}.html"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        logger.error(f"Received HTTP {resp.status_code}: {series_code}. Error logged: {debug_file}")
        return False

    if "<generic:Obs>" not in resp.text and "ObsValue" not in resp.text:
        logger.warning(f"No data returned (no observations found): {series_code}")
        return False

    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bronze_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    n_obs = len(re.findall(r"<generic:Obs>", resp.text))
    logger.info(f"Saved: {bronze_path} ({n_obs} observations)")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="TUIK SDMX collector")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if bronze cache exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not SERIES:
        raise SystemExit("config/tuik_series.py is empty - register a series first.")

    results = {"success": 0, "failed": 0}
    for series in SERIES:
        ok = fetch_series(series["dataflow_id"], series["version"], series["key"], series["code"], force=args.force)
        results["success" if ok else "failed"] += 1

    logger.info("=" * 50)
    logger.info("TUIK FETCH SUMMARY")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Failed : {results['failed']}")
    logger.info("=" * 50)
