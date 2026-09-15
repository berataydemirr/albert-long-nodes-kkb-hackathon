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
import xml.etree.ElementTree as ET

import requests

from config.settings import TUIK_API_KEY, TUIK_BRONZE_DIR, DEBUG_DIR, FULL_START_YEAR, FULL_START_MONTH, FULL_END_YEAR, FULL_END_MONTH
from config.tuik_series import SERIES
from src.common.logging_setup import get_logger

logger = get_logger("tuik")

TOKEN_URL = "https://giris.tuik.gov.tr/realms/web/protocol/openid-connect/token"
DATA_BASE_URL = "https://nsiws.tuik.gov.tr/rest/data/TR"
DATAFLOW_LIST_URL = "https://nsiws.tuik.gov.tr/rest/dataflow/TR/all/latest"
_SDMX_NS = {
    "structure": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "common": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


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


_DATAFLOW_CATALOG_CACHE: list[dict] | None = None


def list_dataflows(refresh: bool = False) -> list[dict]:
    """TUIK's full dataflow catalog - id, version, and Turkish/English
    name for every one of TUIK's ~427 published dataflows. Metadata only:
    this endpoint (confirmed live, standard SDMX 2.1 structure query)
    never returns actual observation data, so listing/searching it
    carries none of the "guessed a key, got a real-but-wrong series"
    risk that fetching data by an unresolved key would.

    Cached per process (the catalog is ~450KB and does not change during
    a session) - pass refresh=True to refetch."""
    global _DATAFLOW_CATALOG_CACHE
    if _DATAFLOW_CATALOG_CACHE is not None and not refresh:
        logger.debug("list_dataflows: served from cache")
        return _DATAFLOW_CATALOG_CACHE

    token = get_access_token()
    resp = requests.get(
        DATAFLOW_LIST_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/xml"},
        timeout=30,
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    dataflows = []
    for df in root.iter(f"{{{_SDMX_NS['structure']}}}Dataflow"):
        name_tr_el = df.find(f"{{{_SDMX_NS['common']}}}Name[@{{{_SDMX_NS['xml']}}}lang='tr']")
        name_en_el = df.find(f"{{{_SDMX_NS['common']}}}Name[@{{{_SDMX_NS['xml']}}}lang='en']")
        dataflows.append({
            "dataflow_id": df.get("id"),
            "version": df.get("version"),
            "name_tr": name_tr_el.text if name_tr_el is not None else "",
            "name_en": name_en_el.text if name_en_el is not None else "",
        })
    logger.info(f"list_dataflows: fetched {len(dataflows)} dataflow(s)")
    _DATAFLOW_CATALOG_CACHE = dataflows
    return dataflows


# Python's plain str.lower() is not Turkish-locale-aware: 'İ'.lower()
# produces 'i' followed by a COMBINING DOT ABOVE (U+0307), two
# codepoints, not the plain ASCII 'i' a query would type - so a search
# for "işgücü" against a catalog name containing "İşgücü" silently found
# nothing, even though it is an exact match by eye. Found live: the
# real registered series' own dataflow is titled "Temel İşgücü
# Göstergeleri", and searching for its own domain word returned zero
# results. Swapping the Turkish capital dotted/dotless I forms first
# fixes both directions before the generic casefold runs.
_TR_CASEFOLD_MAP = str.maketrans({"İ": "i", "I": "ı"})


def _tr_casefold(s: str) -> str:
    return s.translate(_TR_CASEFOLD_MAP).lower()


def search_dataflows(query: str, limit: int = 10) -> list[dict]:
    """Case-insensitive (Turkish-aware) substring match against every
    dataflow's Turkish AND English name - discovery only. A match here is
    the FIRST step of the manual dataflow -> DSD -> codelist key-
    resolution process this project's TUIK integration always requires
    (see config/tuik_series.py module docstring for a worked example,
    including one - DF_DOGUM_KDH_C - that was investigated and abandoned
    because the key could not be confirmed). Finding a matching
    dataflow_id does NOT mean the series can be fetched: fetch_series
    still needs a resolved 'key' that only a human, reading that
    dataflow's own DSD codelists, can produce safely."""
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string.")
    q = _tr_casefold(query.strip())
    matches = [
        d for d in list_dataflows()
        if q in _tr_casefold(d["name_tr"]) or q in _tr_casefold(d["name_en"]) or q in _tr_casefold(d["dataflow_id"])
    ]
    logger.info(f"search_dataflows: query={query!r} -> {len(matches)} match(es), returning up to {limit}")
    return matches[:limit]


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
