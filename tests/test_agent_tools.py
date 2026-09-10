import pytest

from src.agent.tools import (
    anomaly_analysis, change_detection, correlation_analysis,
    causality_analysis, threshold_filter_compare,
)


def _periods(n: int) -> list[str]:
    return [f"2021-{m:02d}" if m <= 12 else f"2022-{m - 12:02d}" for m in range(1, n + 1)]


def test_anomaly_analysis_finds_outlier():
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03", "2021-04"], [10, 11, 10, 100])
    assert result["anomalies"][0]["period"] == "2021-04"


def test_change_detection_orders_by_relative_change():
    result = change_detection(["2021-01", "2021-02", "2021-03"], [100, 110, 55])
    assert result["largest_changes"][0]["period"] == "2021-03"


def test_correlation_warns_against_causal_claim():
    result = correlation_analysis(["2021-01", "2021-02", "2021-03"], [1, 2, 3], ["2021-01", "2021-02", "2021-03"], [2, 4, 6])
    assert result["pearson_correlation"] == pytest.approx(1.0)
    assert "not proof of causality" in result["warning"]


def test_causality_analysis_finds_the_true_lag():
    import math
    import random
    rng = random.Random(42)
    n = 30
    periods = _periods(n)
    x_values = [math.sin(i / 3) for i in range(n)]
    # y is x lagged by 2 periods plus real noise (not near-zero) - a near-perfect
    # copy makes the Granger VAR singular (statsmodels then refuses to fit).
    y_values = [x_values[max(i - 2, 0)] + rng.uniform(-0.15, 0.15) for i in range(n)]

    result = causality_analysis(periods, x_values, periods, y_values, max_lag=4)

    assert result["best_lag"]["lag"] == 2
    assert len(result["granger_pvalues"]) == 4
    assert "not proof" in result["warning"].lower()


def test_causality_analysis_rejects_too_few_periods():
    with pytest.raises(ValueError):
        causality_analysis(["2021-01", "2021-02"], [1, 2], ["2021-01", "2021-02"], [1, 2], max_lag=4)


def test_threshold_filter_compare_separates_window():
    periods = _periods(10)
    filter_values = [1, 1, 1, 5, 5, 5, 1, 1, 1, 1]  # "in window" when between 4 and 6
    compare_values = [10, 11, 12, 13, 20, 30, 31, 32, 33, 34]

    result = threshold_filter_compare(periods, filter_values, periods, compare_values, low=4, high=6)

    assert result["periods_in_window"] == 3
    assert result["avg_change_in_window"] is not None
    assert "descriptive contrast" in result["warning"]
