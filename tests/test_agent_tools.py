import pytest

from src.agent.tools import anomaly_analysis, change_detection, correlation_analysis


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
