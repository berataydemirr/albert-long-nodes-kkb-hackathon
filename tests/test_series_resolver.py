"""Tests for resolving a series by Seri_Kod straight out of the gold layer.

These run against the REAL built database when it exists, because the whole
point of the resolver is that it reads actual rows rather than whatever a
fixture claims. When the database has not been built they skip instead of
failing, so a fresh clone can still run the suite.
"""

import pytest

from src.agent.series_resolver import DB_PATH, resolve_series

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


def _any_seri_kod() -> str:
    """Pick a real code from the catalog so the test does not hard-code one
    that a later gold rebuild might rename."""
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(
            "SELECT Seri_Kod FROM gold_fact_long GROUP BY Seri_Kod ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()


def test_resolve_series_returns_aligned_periods_and_values():
    result = resolve_series(_any_seri_kod())

    assert result["row_count"] > 0
    assert len(result["periods"]) == len(result["values"]) == result["row_count"]
    assert all(isinstance(value, float) for value in result["values"])
    assert all(isinstance(period, str) for period in result["periods"])


def test_resolve_series_orders_periods_ascending():
    result = resolve_series(_any_seri_kod())
    assert result["periods"] == sorted(result["periods"])


def test_resolve_series_respects_period_bounds():
    seri_kod = _any_seri_kod()
    everything = resolve_series(seri_kod)
    midpoint = everything["periods"][len(everything["periods"]) // 2]

    bounded = resolve_series(seri_kod, start_period=midpoint)

    assert bounded["row_count"] < everything["row_count"]
    assert min(bounded["periods"]) >= midpoint


def test_unknown_seri_kod_raises_with_suggestions():
    """The error has to be actionable: a bare 'not found' costs the model a
    whole extra round re-reading the catalog it already had."""
    with pytest.raises(ValueError) as excinfo:
        resolve_series("TOTALLY_MADE_UP_CODE_XYZ")

    message = str(excinfo.value)
    assert "Unknown seri_kod" in message
    assert "Did you mean" in message


def test_filters_that_match_nothing_are_distinguished_from_a_bad_code():
    """An existing code with an impossible period filter is a different
    problem from a nonexistent code, and the model needs to be able to tell
    them apart - one means 'fix the filter', the other 'fix the code'."""
    with pytest.raises(ValueError) as excinfo:
        resolve_series(_any_seri_kod(), start_period="2999-01")

    message = str(excinfo.value)
    assert "exists but no rows match" in message


@pytest.mark.parametrize("bad_input", [None, "", "   ", 42, ["a"]])
def test_rejects_non_string_seri_kod(bad_input):
    with pytest.raises(ValueError):
        resolve_series(bad_input)


def test_rejects_absurdly_long_seri_kod():
    with pytest.raises(ValueError, match="too long"):
        resolve_series("X" * 500)


@pytest.mark.parametrize("bad_period", ["not-a-period", "20-1", "2021-13-01", "'; DROP TABLE gold_fact_long; --"])
def test_rejects_malformed_period_bounds(bad_period):
    with pytest.raises(ValueError):
        resolve_series(_any_seri_kod(), start_period=bad_period)


def test_seri_kod_is_passed_as_a_parameter_not_concatenated():
    """A SQL-injection-shaped Seri_Kod must come back as 'unknown code',
    never execute. If it were interpolated into the query this would raise
    a duckdb parser error (or worse) instead of a clean ValueError."""
    with pytest.raises(ValueError) as excinfo:
        resolve_series("' OR 1=1 --")

    assert "Unknown seri_kod" in str(excinfo.value)
