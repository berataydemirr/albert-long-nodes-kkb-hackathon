"""Read-only data access for the FastAPI backend - pulls the same gold
tables the Streamlit dashboard and the snapshot report use, as plain
JSON-serializable dicts (no Plotly Figure objects here; the Next.js
frontend does its own charting).

Every function is defensive: a missing gold DB or an empty/missing
table returns an empty structure, never raises, so one missing table
doesn't 500 the whole /api/dashboard response.
"""
import math

import duckdb
import pandas as pd

from config.settings import GOLD_DIR
from src.agent.tools import causality_analysis, change_detection
from src.report.charts import query_series_from_gold
from src.common.logging_setup import get_logger

logger = get_logger("api_gold_queries")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"


def _json_safe(value):
    """Recursively converts one value to something Python's json module
    (used by Starlette's JSONResponse with allow_nan=False) will accept.

    Why df.where(df.notna(), None) alone isn't enough: duckdb -> pandas
    can land a NULL as a raw float('nan') inside a pandas 'str'-dtype
    column (confirmed live on gold_fact_long.Birim, which is NULL for 3
    series) - pandas' own .notna() on that dtype doesn't recognize a
    bare float NaN as missing, so .where() leaves it untouched and it
    reaches json.dumps still a NaN, which raises 'Out of range float
    values are not JSON compliant'. Checking isinstance(v, float) +
    math.isnan(v) directly sidesteps the dtype's own (mis)classification
    of what counts as missing. numpy scalars (int64/float64, e.g. from
    causality_analysis's Granger test keys or a duckdb integer column)
    are unwrapped via .item() for the same reason - FastAPI's default
    encoder doesn't know what to do with them either."""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if hasattr(value, "item") and not isinstance(value, (dict, list)):  # numpy scalar
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts. See _json_safe() for why a
    plain .where(df.notna(), None) isn't reliable enough on its own."""
    if df.empty:
        return []
    return [_json_safe(row) for row in df.to_dict("records")]


def _table_exists(con, name: str) -> bool:
    return name in {row[0] for row in con.execute("SHOW TABLES").fetchall()}


def get_konut_demo() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if not _table_exists(con, "gold_konut_demo"):
            return []
        return _records(con.execute("SELECT * FROM gold_konut_demo ORDER BY Period").fetchdf())
    finally:
        con.close()


def get_series_catalog() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if not _table_exists(con, "gold_fact_long"):
            return []
        return _records(con.execute(
            "SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim, Tip FROM gold_fact_long ORDER BY Kaynak, Seri_Kod"
        ).fetchdf())
    finally:
        con.close()


def get_health() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if not _table_exists(con, "gold_health_life_expectancy"):
            return []
        return _records(con.execute("SELECT * FROM gold_health_life_expectancy ORDER BY Period").fetchdf())
    finally:
        con.close()


def _quarterly_sort_key(period: str) -> tuple[int, int]:
    """'YYYY-Q' (Q in 3,6,9,12) -> (year, quarter). SQL's MAX(Period) /
    plain string comparison is WRONG here since these periods aren't
    zero-padded: '2021-12' < '2021-9' lexicographically, even though Q4
    comes after Q3. Confirmed live: MAX(Period) happened to return the
    right 'latest quarter' today only because the most recent year in
    the data doesn't have a Q4 yet - it silently breaks (returns Q3
    instead of Q4) the moment one is added."""
    year_str, q_str = period.split("-")
    return (int(year_str), int(q_str))


def get_regional_top(period: str | None = None, limit: int = 10) -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if not _table_exists(con, "gold_finturk_regional_quarterly"):
            return []
        if period is None:
            available = [row[0] for row in con.execute("SELECT DISTINCT Period FROM gold_finturk_regional_quarterly").fetchall()]
            if not available:
                return []
            period = max(available, key=_quarterly_sort_key)
        df = con.execute(
            "SELECT Bolge, Toplam_Nakdi_Kredi_Bin_TL FROM gold_finturk_regional_quarterly "
            "WHERE Period = ? ORDER BY Toplam_Nakdi_Kredi_Bin_TL DESC LIMIT ?",
            [period, limit],
        ).fetchdf()
        return _records(df)
    finally:
        con.close()


def get_net_profit() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if not _table_exists(con, "gold_bddk_net_profit_monthly"):
            return []
        return _records(con.execute("SELECT Period, Value_Monthly FROM gold_bddk_net_profit_monthly ORDER BY Period").fetchdf())
    finally:
        con.close()


def get_causality_demo() -> dict | None:
    """Faiz orani -> konut kredisi, hackathon brief'inin vitrin ornegi."""
    faiz = query_series_from_gold("TP.BKR.TRY.18")
    konut = query_series_from_gold("tablo_4.konut_kredisi")
    if faiz.empty or konut.empty or len(faiz) < 17 or len(konut) < 17:
        logger.warning("get_causality_demo: insufficient data")
        return None
    try:
        return causality_analysis(
            faiz["Period"].tolist(), faiz["Deger"].tolist(),
            konut["Period"].tolist(), konut["Deger"].tolist(),
            max_lag=6,
        )
    except Exception:
        logger.exception("get_causality_demo: causality_analysis failed")
        return None


def get_policy_rate_with_changes() -> dict:
    df = query_series_from_gold("TP.BISPOLFAIZ.TUR")
    if df.empty:
        return {"series": [], "change_detection": {"largest_changes": []}}
    try:
        changes = change_detection(df["Period"].tolist(), df["Deger"].tolist())
    except Exception:
        logger.exception("get_policy_rate_with_changes: change_detection failed")
        changes = {"largest_changes": []}
    return {"series": _records(df), "change_detection": changes}


def get_dashboard_snapshot() -> dict:
    """Everything /api/dashboard needs, gathered in one call - avoids the
    frontend making 6 separate requests on page load. _json_safe() wraps
    the WHOLE result as a final safety net (not just each panel
    individually) so a field added later without remembering to clean it
    still can't reach json.dumps as a raw NaN/numpy scalar."""
    logger.info("get_dashboard_snapshot: gathering all panels")
    return _json_safe({
        "konut_demo": get_konut_demo(),
        "catalog": get_series_catalog(),
        "health": get_health(),
        "regional_top": get_regional_top(),
        "net_profit": get_net_profit(),
        "causality": get_causality_demo(),
        "policy_rate": get_policy_rate_with_changes(),
        "gold_db_exists": DB_PATH.exists(),
    })
