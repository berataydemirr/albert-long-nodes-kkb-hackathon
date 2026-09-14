"""Tests for the Seri_Kod input path on the analysis tools.

The contract under test: passing a Seri_Kod must produce the same numbers
as passing the arrays by hand, because the whole optimization rests on the
two being interchangeable.
"""

import pytest

from src.agent.series_resolver import DB_PATH, resolve_series
from src.agent.tools import (
    anomaly_analysis,
    causality_analysis,
    change_detection,
    correlation_analysis,
    threshold_filter_compare,
)

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


def _two_long_series() -> tuple[str, str]:
    """Two codes with enough history for a lag test."""
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = con.execute(
            "SELECT Seri_Kod FROM gold_fact_long GROUP BY Seri_Kod "
            "HAVING COUNT(*) >= 20 ORDER BY COUNT(*) DESC, Seri_Kod LIMIT 2"
        ).fetchall()
    finally:
        con.close()
    if len(rows) < 2:
        pytest.skip("need two series with at least 20 periods")
    return rows[0][0], rows[1][0]


def test_seri_kod_and_arrays_give_identical_anomaly_results():
    seri_kod, _ = _two_long_series()
    resolved = resolve_series(seri_kod)

    by_code = anomaly_analysis(seri_kod=seri_kod)
    by_arrays = anomaly_analysis(periods=resolved["periods"], values=resolved["values"])

    assert by_code["anomalies"] == by_arrays["anomalies"]


def test_seri_kod_and_arrays_give_identical_change_results():
    seri_kod, _ = _two_long_series()
    resolved = resolve_series(seri_kod)

    by_code = change_detection(seri_kod=seri_kod)
    by_arrays = change_detection(periods=resolved["periods"], values=resolved["values"])

    assert by_code["largest_changes"] == by_arrays["largest_changes"]


def test_seri_kod_and_arrays_give_identical_correlation():
    x_kod, y_kod = _two_long_series()
    x, y = resolve_series(x_kod), resolve_series(y_kod)

    by_code = correlation_analysis(x_seri_kod=x_kod, y_seri_kod=y_kod)
    by_arrays = correlation_analysis(
        x_periods=x["periods"], x_values=x["values"],
        y_periods=y["periods"], y_values=y["values"],
    )

    assert by_code["pearson_correlation"] == by_arrays["pearson_correlation"]
    assert by_code["shared_periods"] == by_arrays["shared_periods"]


def test_causality_accepts_seri_kod_pair():
    x_kod, y_kod = _two_long_series()

    result = causality_analysis(x_seri_kod=x_kod, y_seri_kod=y_kod, max_lag=3)

    assert result["shared_periods"] > 0
    assert result["x_source"]["seri_kod"] == x_kod
    assert result["y_source"]["seri_kod"] == y_kod
    assert "not proof of true causation" in result["warning"]


def test_threshold_filter_compare_accepts_seri_kod_pair():
    filter_kod, compare_kod = _two_long_series()
    filter_series = resolve_series(filter_kod)
    low = min(filter_series["values"])
    high = sorted(filter_series["values"])[len(filter_series["values"]) // 2]

    result = threshold_filter_compare(
        filter_seri_kod=filter_kod, compare_seri_kod=compare_kod, low=low, high=high
    )

    assert result["periods_in_window"] > 0
    assert result["filter_source"]["seri_kod"] == filter_kod


def test_result_reports_where_the_numbers_came_from():
    """Source metadata is what lets the final answer cite the series and
    period range without the model having to remember them."""
    seri_kod, _ = _two_long_series()

    result = anomaly_analysis(seri_kod=seri_kod)

    source = result["source"]
    assert source["source"] == "lakehouse"
    assert source["seri_kod"] == seri_kod
    assert source["first_period"] <= source["last_period"]


def test_mixing_seri_kod_and_arrays_is_rejected():
    """Silently preferring one side would make a wrong answer impossible
    to explain afterwards."""
    seri_kod, _ = _two_long_series()

    with pytest.raises(ValueError, match="not both"):
        anomaly_analysis(seri_kod=seri_kod, periods=["2021-01"], values=[1.0])


def test_supplying_neither_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError) as excinfo:
        anomaly_analysis()

    assert "seri_kod" in str(excinfo.value)


def test_period_bounds_narrow_the_analysed_window():
    seri_kod, _ = _two_long_series()
    everything = resolve_series(seri_kod)
    midpoint = everything["periods"][len(everything["periods"]) // 2]

    narrowed = anomaly_analysis(seri_kod=seri_kod, start_period=midpoint)

    assert narrowed["source"]["row_count"] < everything["row_count"]


def test_unknown_seri_kod_surfaces_suggestions_through_the_tool():
    with pytest.raises(ValueError, match="Did you mean"):
        anomaly_analysis(seri_kod="NO_SUCH_SERIES")


def test_causality_error_names_a_usable_lag():
    """The old message just refused; the model then guessed another lag and
    burned a round per guess."""
    periods = [f"2021-{i:02d}" for i in range(1, 11)]
    values = [float(i) for i in range(10)]

    with pytest.raises(ValueError, match="max_lag="):
        causality_analysis(
            x_periods=periods, x_values=values,
            y_periods=periods, y_values=values, max_lag=6,
        )


def test_threshold_filter_compare_still_rejects_non_numeric_bounds():
    seri_kod, other = _two_long_series()

    with pytest.raises(ValueError, match="must both be numbers"):
        threshold_filter_compare(filter_seri_kod=seri_kod, compare_seri_kod=other, low="dusuk", high=10)
