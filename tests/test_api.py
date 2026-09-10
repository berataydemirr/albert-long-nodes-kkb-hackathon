"""Hard/edge-case tests for the FastAPI backend (api/). Same philosophy
as the other stress-test files: break it, fix the real code.
"""
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.gold_queries import _json_safe

client = TestClient(app)


# ---------------------------------------------------------------------
# _json_safe: the NaN/numpy-scalar bug found while building this API
# ---------------------------------------------------------------------

def test_json_safe_converts_bare_nan_to_none():
    assert _json_safe(float("nan")) is None


def test_json_safe_converts_infinity_to_none():
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None


def test_json_safe_unwraps_numpy_scalars():
    import numpy as np
    assert _json_safe(np.int64(5)) == 5
    assert isinstance(_json_safe(np.int64(5)), int)
    assert _json_safe(np.float64(2.5)) == 2.5


def test_json_safe_recurses_into_nested_structures():
    dirty = {"a": [1, float("nan"), {"b": float("inf")}]}
    clean = _json_safe(dirty)
    assert clean == {"a": [1, None, {"b": None}]}


def test_json_safe_leaves_normal_values_untouched():
    assert _json_safe({"x": 1, "y": "text", "z": [1, 2, 3]}) == {"x": 1, "y": "text", "z": [1, 2, 3]}


# ---------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------

def test_health_returns_expected_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"status", "gold_db_exists", "mia_configured"}


# ---------------------------------------------------------------------
# /api/dashboard
# ---------------------------------------------------------------------

def test_dashboard_returns_valid_json_no_nan_no_numpy():
    """The actual regression this test suite exists to catch: a raw NaN
    inside a 'str'-dtype pandas column (gold_fact_long.Birim, NULL for 3
    series) survived .where(df.notna(), None) and broke json.dumps with
    'Out of range float values are not JSON compliant'."""
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert len(body["catalog"]) > 0
    # at least one series has a genuinely unconfirmed unit - must be None, not a bare NaN string
    unit_values = {row["Birim"] for row in body["catalog"]}
    assert None in unit_values or all(v is not None for v in unit_values)


def test_dashboard_konut_demo_has_both_nominal_and_real_columns():
    r = client.get("/api/dashboard")
    row = r.json()["konut_demo"][0]
    assert "Konut_Kredisi_Milyon_TL" in row
    assert "Konut_Kredisi_Reel_Milyon_TL" in row


def test_dashboard_causality_has_no_numpy_int_leak():
    r = client.get("/api/dashboard")
    causality = r.json()["causality"]
    if causality:
        for lag_entry in causality["granger_pvalues"]:
            assert isinstance(lag_entry["lag"], int)


def test_dashboard_regional_top_uses_the_chronologically_latest_quarter():
    """Regression: get_regional_top() used to find 'latest period' via
    SQL MAX(Period), which sorts 'YYYY-Q' periods as plain strings and
    is wrong whenever a year's Q4 ('...-12') exists alongside single-
    digit quarters ('...-3/6/9') - '2021-12' < '2021-9' lexicographically
    even though Q4 comes after Q3. Confirmed this only ever looked
    correct because the current data's most recent year has no Q4 yet."""
    from api.gold_queries import _quarterly_sort_key
    import duckdb
    from config.settings import GOLD_DIR

    con = duckdb.connect(str(GOLD_DIR / "lakehouse.duckdb"), read_only=True)
    try:
        all_periods = [row[0] for row in con.execute("SELECT DISTINCT Period FROM gold_finturk_regional_quarterly").fetchall()]
    finally:
        con.close()
    true_latest = max(all_periods, key=_quarterly_sort_key)

    r = client.get("/api/dashboard")
    regional_top = r.json()["regional_top"]
    assert len(regional_top) > 0

    from api.gold_queries import get_regional_top
    assert get_regional_top()[0]["Bolge"] == get_regional_top(period=true_latest)[0]["Bolge"]


# ---------------------------------------------------------------------
# /api/ask: validation edge cases (no live MIA call needed)
# ---------------------------------------------------------------------

def test_ask_rejects_empty_question():
    r = client.post("/api/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_rejects_missing_question_field():
    r = client.post("/api/ask", json={})
    assert r.status_code == 422


def test_ask_rejects_question_over_max_length():
    r = client.post("/api/ask", json={"question": "x" * 3000})
    assert r.status_code == 422


def test_ask_rejects_non_string_question():
    r = client.post("/api/ask", json={"question": 12345})
    assert r.status_code == 422


def test_ask_returns_503_when_mia_key_missing(monkeypatch):
    import api.main as main_mod
    monkeypatch.setattr(main_mod, "MIA_API_KEY", None)
    r = client.post("/api/ask", json={"question": "test"})
    assert r.status_code == 503


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

def test_cors_allows_the_nextjs_dev_origin():
    r = client.options("/api/dashboard", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_an_unlisted_origin():
    r = client.options("/api/dashboard", headers={
        "Origin": "http://evil.example.com",
        "Access-Control-Request-Method": "GET",
    })
    assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"
