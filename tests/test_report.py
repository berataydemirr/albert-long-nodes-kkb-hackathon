"""Hard/edge-case tests for Faz D (src/report/): citation extraction,
chart generation, HTML report export. Same philosophy as
test_agent_tools_stress.py - break it, fix the real code, not the test.
"""
import json

import pytest

from src.report.citations import extract_citations, format_citations_markdown
from src.report.charts import series_line_chart, causality_lag_chart, anomaly_chart, query_series_from_gold
from src.report.export import build_report_html, save_report


# ---------------------------------------------------------------------
# citations.py
# ---------------------------------------------------------------------

def test_empty_trace_is_flagged_ungrounded():
    result = extract_citations([])
    assert result["grounded"] is False
    assert "cagirmadan" in result["warning"]


def test_schema_only_trace_is_still_ungrounded():
    """Calling lakehouse_schema (a discovery call) without ever querying
    data must NOT count as a grounded citation."""
    trace = [{"round": 1, "name": "lakehouse_schema", "arguments": "{}", "error": None}]
    result = extract_citations(trace)
    assert result["grounded"] is False


def test_lakehouse_query_citation_extracts_table_and_series():
    trace = [{
        "round": 1, "name": "lakehouse_query",
        "arguments": json.dumps({"sql": "SELECT Deger FROM gold_fact_long WHERE Seri_Kod = 'TP.BKR.TRY.18'"}),
        "error": None,
    }]
    result = extract_citations(trace)
    assert result["grounded"] is True
    assert result["citations"][0]["tables"] == ["gold_fact_long"]
    assert result["citations"][0]["seri_kodlar"] == ["TP.BKR.TRY.18"]


def test_failed_tool_call_is_reported_not_hidden():
    trace = [{"round": 1, "name": "lakehouse_query", "arguments": "{}", "error": "Unknown table(s): ['x']"}]
    result = extract_citations(trace)
    assert result["grounded"] is False
    assert len(result["tool_failures"]) == 1
    assert result["tool_failures"][0]["tool"] == "lakehouse_query"


def test_malformed_arguments_json_does_not_crash_extraction():
    trace = [{"round": 1, "name": "lakehouse_query", "arguments": "{not valid json", "error": None}]
    result = extract_citations(trace)  # must not raise
    assert isinstance(result, dict)


def test_arguments_none_does_not_crash_extraction():
    trace = [{"round": 1, "name": "causality_analysis", "arguments": None, "error": None}]
    result = extract_citations(trace)
    assert result["grounded"] is True


def test_duplicate_citations_are_deduplicated():
    entry = {
        "round": 1, "name": "lakehouse_query",
        "arguments": json.dumps({"sql": "SELECT Deger FROM gold_fact_long WHERE Seri_Kod = 'TP.KFE.TR'"}),
        "error": None,
    }
    result = extract_citations([entry, dict(entry, round=2)])
    assert len(result["citations"]) == 1


def test_lakehouse_query_citation_extracts_series_from_in_clause():
    """A comparison query ('faiz ile kredi karsilastir') typically uses
    Seri_Kod IN (...), not a single '=' - both series must be cited."""
    trace = [{
        "round": 1, "name": "lakehouse_query",
        "arguments": json.dumps({"sql": "SELECT * FROM gold_fact_long WHERE Seri_Kod IN ('TP.BKR.TRY.18', 'TP.KFE.TR')"}),
        "error": None,
    }]
    result = extract_citations(trace)
    assert result["citations"][0]["seri_kodlar"] == ["TP.BKR.TRY.18", "TP.KFE.TR"]


def test_stat_tool_citation_has_no_table_reference():
    trace = [{"round": 1, "name": "correlation_analysis", "arguments": "{}", "error": None}]
    result = extract_citations(trace)
    assert result["citations"][0]["tool"] == "correlation_analysis"


def test_format_citations_markdown_handles_ungrounded_result():
    md = format_citations_markdown(extract_citations([]))
    assert "Kaynak yok" in md


# ---------------------------------------------------------------------
# charts.py
# ---------------------------------------------------------------------

def test_series_line_chart_empty_input_does_not_raise():
    fig = series_line_chart([], [], "Bos Grafik")
    assert fig is not None


def test_series_line_chart_mismatched_lengths_does_not_raise():
    fig = series_line_chart(["2021-01", "2021-02"], [1.0], "Uyusmayan Uzunluk")
    assert fig is not None


def test_series_line_chart_all_none_values_does_not_raise():
    fig = series_line_chart(["2021-01", "2021-02"], [None, None], "Hepsi Bos")
    assert fig is not None


def test_series_line_chart_single_point_does_not_raise():
    fig = series_line_chart(["2021-01"], [42.0], "Tek Nokta")
    assert len(fig.data) == 1


def test_causality_lag_chart_empty_lag_correlations_does_not_raise():
    fig = causality_lag_chart({"lag_correlations": [], "best_lag": None})
    assert fig is not None


def test_causality_lag_chart_missing_keys_does_not_raise():
    fig = causality_lag_chart({})  # neither key present
    assert fig is not None


def test_anomaly_chart_anomaly_period_not_in_series_does_not_raise():
    """Defensive: an anomaly_result referencing a period that isn't in
    the periods list (e.g. mismatched inputs from a caller bug) must not
    crash chart rendering."""
    fig = anomaly_chart(["2021-01", "2021-02"], [1.0, 2.0], {"anomalies": [{"period": "1999-01", "value": 999, "score": 9}]})
    assert fig is not None


def test_anomaly_chart_mismatched_lengths_does_not_raise():
    fig = anomaly_chart(["2021-01"], [1.0, 2.0], {"anomalies": []})
    assert fig is not None


def test_query_series_from_gold_unknown_series_returns_empty_not_error():
    df = query_series_from_gold("THIS_SERIES_DOES_NOT_EXIST")
    assert df.empty


def test_query_series_from_gold_real_series_returns_rows():
    df = query_series_from_gold("TP.BKR.TRY.18")
    assert len(df) > 0


# ---------------------------------------------------------------------
# export.py
# ---------------------------------------------------------------------

def test_build_report_html_with_no_figures():
    html = build_report_html("Soru?", "Cevap.", [])
    assert "<html" in html.lower()


def test_build_report_html_escapes_html_in_answer():
    """The answer text comes from an LLM - it must never be able to
    inject raw HTML/script into the report."""
    html = build_report_html("Soru?", "<script>alert(1)</script>", [])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_report_html_handles_turkish_characters():
    html = build_report_html("Konut kredisi faiz oranı nasıl değişti?", "Değişim İSTANBUL'da gözlendi.", [])
    assert "değişti" in html
    assert "İSTANBUL" in html


def test_build_report_html_with_charts():
    fig = series_line_chart(["2021-01", "2021-02"], [1.0, 2.0], "Test")
    html = build_report_html("Soru?", "Cevap.", [], figures=[fig])
    assert "plotly" in html.lower()


def test_build_report_html_empty_question_and_answer_do_not_crash():
    html = build_report_html("", "", [])
    assert "<html" in html.lower()


def test_save_report_writes_a_file(tmp_path, monkeypatch):
    import src.report.export as export_mod
    monkeypatch.setattr(export_mod, "REPORTS_DIR", tmp_path)
    path = save_report("<html><body>test</body></html>", filename="test_report")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "<html><body>test</body></html>"


def test_save_report_default_filename_has_no_extension_collision(tmp_path, monkeypatch):
    import src.report.export as export_mod
    monkeypatch.setattr(export_mod, "REPORTS_DIR", tmp_path)
    path = save_report("<html></html>")
    assert path.suffix == ".html"
