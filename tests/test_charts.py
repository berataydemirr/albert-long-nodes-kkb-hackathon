"""Tests for chart derivation from tool results.

The audience for the UI has basic financial literacy, so a wrong or
misleading chart is worse than no chart - most of these tests are about
what must NOT be drawn, not what must.
"""

import pytest

from src.agent.charts import (
    MAX_BARS,
    MAX_CHARTS_PER_ANSWER,
    MIN_POINTS_FOR_LINE,
    derive_charts,
)
from src.agent.series_resolver import DB_PATH

needs_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


def _call(name, output_json, **extra):
    return {"name": name, "output_json": output_json, "error": None, **extra}


# --- compare_regions -> horizontal bar ----------------------------------

def test_region_ranking_becomes_a_bar_chart():
    charts = derive_charts([_call("compare_regions", {
        "period": "2026-6", "metric": "npl", "birim": "%",
        "ranked": [
            {"rank": 1, "bolge": "MARDİN", "value": 9.29},
            {"rank": 2, "bolge": "KARAMAN", "value": 7.77},
        ],
    })])

    assert len(charts) == 1
    assert charts[0]["kind"] == "bar"
    assert charts[0]["unit"] == "%"
    assert [b["label"] for b in charts[0]["bars"]] == ["MARDİN", "KARAMAN"]


def test_bar_count_is_capped():
    ranked = [{"rank": i, "bolge": f"IL{i}", "value": 100 - i} for i in range(1, 40)]
    charts = derive_charts([_call("compare_regions", {"period": "2026-6", "metric": "kredi", "ranked": ranked})])

    assert len(charts[0]["bars"]) <= MAX_BARS


def test_growth_is_preferred_over_share_as_the_bar_note():
    """When a comparison period was asked for, the change is the story -
    the share of the national total is the less interesting number."""
    charts = derive_charts([_call("compare_regions", {
        "period": "2026-6", "metric": "kredi", "compare_to_period": "2025-6",
        "ranked": [{"rank": 1, "bolge": "İSTANBUL", "value": 100, "share_pct": 34.0, "growth_pct": 32.5}],
    })])

    assert "32,5" in charts[0]["bars"][0]["note"].replace(".", ",")


def test_empty_ranking_produces_no_chart():
    assert derive_charts([_call("compare_regions", {"period": "2026-6", "metric": "kredi", "ranked": []})]) == []


# --- lakehouse_query -> line, but only when unambiguous ------------------

def test_period_plus_one_numeric_column_becomes_a_line():
    rows = [{"Period": f"2021-{m:02d}", "Deger": float(m)} for m in range(1, 9)]
    charts = derive_charts([_call("lakehouse_query", {"columns": ["Period", "Deger"], "rows": rows})])

    assert len(charts) == 1
    assert charts[0]["kind"] == "line"
    assert len(charts[0]["points"]) == 8


def test_query_without_a_period_column_is_not_charted():
    """Guessing a shape produces a chart that says something the data does
    not - worse than showing none."""
    rows = [{"Bolge": "A", "Deger": 1.0}, {"Bolge": "B", "Deger": 2.0},
            {"Bolge": "C", "Deger": 3.0}, {"Bolge": "D", "Deger": 4.0}]
    assert derive_charts([_call("lakehouse_query", {"columns": ["Bolge", "Deger"], "rows": rows})]) == []


def test_query_with_two_numeric_columns_is_not_charted():
    """Two measures on one plot is the dual-axis mistake; refusing to draw
    is the correct fallback here."""
    rows = [{"Period": f"2021-{m:02d}", "A": float(m), "B": float(m * 2)} for m in range(1, 9)]
    assert derive_charts([_call("lakehouse_query", {"columns": ["Period", "A", "B"], "rows": rows})]) == []


def test_too_few_points_is_not_charted():
    rows = [{"Period": "2021-01", "Deger": 1.0}, {"Period": "2021-02", "Deger": 2.0}]
    assert len(rows) < MIN_POINTS_FOR_LINE
    assert derive_charts([_call("lakehouse_query", {"columns": ["Period", "Deger"], "rows": rows})]) == []


def test_line_points_are_sorted_by_period():
    rows = [{"Period": "2021-03", "Deger": 3.0}, {"Period": "2021-01", "Deger": 1.0},
            {"Period": "2021-04", "Deger": 4.0}, {"Period": "2021-02", "Deger": 2.0}]
    charts = derive_charts([_call("lakehouse_query", {"columns": ["Period", "Deger"], "rows": rows})])

    labels = [p["label"] for p in charts[0]["points"]]
    assert labels == sorted(labels)


# --- analysis tools -> emphasis on a resolved series ---------------------

@needs_db
def test_anomaly_result_draws_the_series_with_flagged_points():
    charts = derive_charts([_call("anomaly_analysis", {
        "source": {"source": "lakehouse", "seri_kod": "TP.BKR.TRY.18"},
        "anomalies": [{"period": "2024-10", "value": 1.0, "score": 4.0}],
    })])

    assert len(charts) == 1
    assert charts[0]["kind"] == "line"
    highlighted = [p["label"] for p in charts[0]["points"] if p["highlight"]]
    assert highlighted == ["2024-10"]


@needs_db
def test_two_series_analysis_draws_two_separate_charts():
    """Never one plot with two y-axes: a dual axis invents a correlation by
    choosing where the scales line up."""
    charts = derive_charts([_call("causality_analysis", {
        "x_source": {"seri_kod": "TP.BKR.TRY.18"},
        "y_source": {"seri_kod": "tablo_4.konut_kredisi"},
    })])

    assert len(charts) == 2
    assert all(c["kind"] == "line" for c in charts)
    assert charts[0]["title"] != charts[1]["title"]


@needs_db
def test_unknown_series_code_is_skipped_not_fatal():
    charts = derive_charts([_call("anomaly_analysis", {
        "source": {"seri_kod": "NO_SUCH_SERIES"}, "anomalies": [],
    })])

    assert charts == []


@needs_db
def test_units_reach_the_chart_spec():
    """A rate chart with no unit label is ambiguous to a non-expert reader,
    which is exactly the audience. The gold layer lost EVDS units at one
    point; this pins that they arrive."""
    charts = derive_charts([_call("anomaly_analysis", {
        "source": {"seri_kod": "TP.BKR.TRY.18"}, "anomalies": [],
    })])

    assert charts[0]["unit"] == "%"


# --- guards -------------------------------------------------------------

def test_failed_and_blocked_calls_are_never_charted():
    bad = [
        {"name": "compare_regions", "error": "boom",
         "output_json": {"period": "2026-6", "metric": "kredi", "ranked": [{"bolge": "X", "value": 1}]}},
        {"name": "compare_regions", "blocked": True, "error": None,
         "output_json": {"period": "2026-6", "metric": "kredi", "ranked": [{"bolge": "Y", "value": 1}]}},
    ]
    assert derive_charts(bad) == []


def test_chart_count_is_capped():
    calls = [
        _call("compare_regions", {
            "period": "2026-6", "metric": f"metrik{i}",
            "ranked": [{"bolge": "X", "value": 1}],
        })
        for i in range(10)
    ]
    assert len(derive_charts(calls)) <= MAX_CHARTS_PER_ANSWER


def test_duplicate_charts_are_collapsed():
    """The model often runs the same analysis twice; one chart of it
    informs, two is noise."""
    same = {"period": "2026-6", "metric": "npl", "ranked": [{"bolge": "MARDİN", "value": 9.29}]}
    charts = derive_charts([_call("compare_regions", same), _call("compare_regions", dict(same))])

    assert len(charts) == 1


def test_non_dict_output_is_ignored():
    assert derive_charts([_call("compare_regions", "not a dict"), _call("lakehouse_query", None)]) == []


def test_malformed_payload_does_not_raise():
    """derive_charts runs after a correct answer already exists - it must
    never be the reason a good answer fails to render."""
    assert derive_charts([_call("compare_regions", {"ranked": [{"no_label_key": 1}]})]) is not None
    assert derive_charts([{"name": "lakehouse_query", "output_json": {"columns": None, "rows": None}}]) == []


def test_empty_trace_produces_no_charts():
    assert derive_charts([]) == []


def test_metric_slug_becomes_a_human_title():
    """'npl' is a fine API parameter and a useless chart title for a reader
    who is not an analyst - and that reader is the target audience."""
    charts = derive_charts([_call("compare_regions", {
        "period": "2026-6", "metric": "npl",
        "ranked": [{"bolge": "MARDİN", "value": 9.29}],
    })])

    assert "npl" not in charts[0]["title"]
    assert "Takipteki alacak oranı" in charts[0]["title"]


def test_unmapped_metric_still_gets_a_readable_title():
    charts = derive_charts([_call("compare_regions", {
        "period": "2026-6", "metric": "yeni_eklenen_metrik",
        "ranked": [{"bolge": "X", "value": 1}],
    })])

    assert "_" not in charts[0]["title"]
