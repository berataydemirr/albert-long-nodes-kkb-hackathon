import pytest

from src.agent.tools import (
    anomaly_analysis, change_detection, correlation_analysis,
    lakehouse_schema, lakehouse_query,
)


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


def test_lakehouse_schema_includes_series_catalog():
    result = lakehouse_schema()
    assert "gold_fact_long" in result["tables"]
    assert "gold_fact_long_series_catalog" in result
    assert len(result["gold_fact_long_series_catalog"]) > 0


def test_lakehouse_query_runs_a_real_select():
    result = lakehouse_query("SELECT Seri_Kod, Deger FROM gold_fact_long WHERE Seri_Kod = 'TP.BKR.TRY.18' ORDER BY Period LIMIT 3")
    assert result["row_count"] == 3
    assert result["columns"] == ["Seri_Kod", "Deger"]


def test_lakehouse_query_rejects_multiple_statements():
    with pytest.raises(ValueError, match="single SELECT"):
        lakehouse_query("SELECT 1; DROP TABLE gold_fact_long")


def test_lakehouse_query_rejects_ddl_keyword():
    with pytest.raises(ValueError, match="read-only SELECT"):
        lakehouse_query("CREATE TABLE x AS SELECT 1 FROM gold_fact_long")


def test_lakehouse_query_rejects_unknown_table():
    with pytest.raises(ValueError, match="Unknown table"):
        lakehouse_query("SELECT * FROM this_table_does_not_exist")
