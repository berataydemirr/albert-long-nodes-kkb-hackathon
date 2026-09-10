"""Adversarial / edge-case stress tests for src/agent/tools.py.

Not the happy-path tests in test_agent_tools.py - this file exists to
break things: SQL injection variants against lakehouse_query's
guardrail, zero-variance/NaN inputs, duplicate periods, malformed JSON
at the execute_tool boundary. Every failure found here was fixed in
tools.py, not worked around in the test.

Run: pytest tests/test_agent_tools_stress.py -v
"""
import json
import math

import pytest

from src.agent.tools import (
    lakehouse_query, execute_tool,
    anomaly_analysis, change_detection, correlation_analysis,
    causality_analysis, threshold_filter_compare,
    describe_series, compare_regions, _quarterly_sort_key,
)


# ---------------------------------------------------------------------
# lakehouse_query: guardrail bypass attempts
# ---------------------------------------------------------------------

def test_blocks_system_table_introspection():
    """DuckDB table functions (not in our allowlist) must be rejected -
    this is how an attacker would try to read filesystem/catalog info."""
    with pytest.raises(ValueError):
        lakehouse_query("SELECT * FROM duckdb_tables()")


def test_blocks_read_csv_file_read():
    with pytest.raises(ValueError):
        lakehouse_query("SELECT * FROM read_csv('/etc/passwd')")


def test_blocks_semicolon_stacked_statement():
    with pytest.raises(ValueError):
        lakehouse_query("SELECT 1; DROP TABLE gold_fact_long")


def test_blocks_ddl_even_without_semicolon():
    with pytest.raises(ValueError):
        lakehouse_query("SELECT * FROM gold_fact_long WHERE 1=1; ALTER TABLE gold_fact_long ADD COLUMN x INT")


def test_case_and_whitespace_do_not_bypass_guardrail():
    with pytest.raises(ValueError):
        lakehouse_query("   \n\t SeLeCt * from gold_fact_long; DrOp TaBlE gold_fact_long")


def test_allows_a_real_query_with_a_sql_comment():
    """A '--' comment containing something that LOOKS dangerous must not
    false-positive the guardrail, since a comment can't execute."""
    result = lakehouse_query("SELECT Seri_Kod FROM gold_fact_long -- ; drop everything\nLIMIT 1")
    assert result["row_count"] <= 1


def test_allows_a_common_table_expression():
    """WITH ... SELECT is extremely common SQL and must not be rejected
    just because the statement doesn't start with the literal word
    'select'."""
    result = lakehouse_query("WITH recent AS (SELECT * FROM gold_fact_long) SELECT COUNT(*) AS n FROM recent")
    assert result["row_count"] == 1


def test_hard_row_cap_is_enforced_even_with_an_explicit_large_limit():
    """A query that explicitly asks for more than the intended cap must
    still come back capped - the cap is a resource/context-size
    guarantee, not just a default for lazy queries."""
    result = lakehouse_query("SELECT * FROM gold_fact_long LIMIT 999999999")
    assert result["row_count"] <= 200


def test_result_is_always_valid_json_serializable():
    result = lakehouse_query("SELECT * FROM gold_fact_long LIMIT 5")
    json.dumps(result, default=str)  # must not raise


def test_empty_sql_string_raises_cleanly_not_a_crash():
    with pytest.raises(ValueError):
        lakehouse_query("")


def test_whitespace_only_sql_raises_cleanly():
    with pytest.raises(ValueError):
        lakehouse_query("   \n\t  ")


# ---------------------------------------------------------------------
# execute_tool: malformed input at the model-facing boundary
# ---------------------------------------------------------------------

def test_execute_tool_survives_invalid_json_arguments():
    with pytest.raises(Exception):
        execute_tool("lakehouse_query", "not valid json{{{")


def test_execute_tool_survives_unknown_tool_name():
    with pytest.raises(ValueError):
        execute_tool("delete_everything", "{}")


def test_execute_tool_survives_missing_required_argument():
    with pytest.raises(Exception):
        execute_tool("lakehouse_query", "{}")  # missing 'sql'


def test_execute_tool_result_is_always_json_parseable_on_success():
    out = execute_tool("lakehouse_query", json.dumps({"sql": "SELECT 1 AS x"}))
    json.loads(out)  # must not raise


# ---------------------------------------------------------------------
# anomaly_analysis / change_detection / correlation_analysis: numeric edge cases
# ---------------------------------------------------------------------

def test_anomaly_analysis_zero_variance_series_no_crash():
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03"], [5.0, 5.0, 5.0])
    assert result["anomalies"] == []


def test_anomaly_analysis_with_none_values_mixed_in():
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03", "2021-04"], [10, 11, None, 100])
    json.dumps(result, default=str)


def test_change_detection_all_zero_values_no_division_error():
    result = change_detection(["2021-01", "2021-02", "2021-03"], [0, 0, 0])
    assert result["largest_changes"] == []


def test_correlation_analysis_zero_variance_x_returns_json_safe_result():
    """A constant series has undefined (NaN) correlation - the result
    must still be valid JSON (None, not a bare NaN token)."""
    result = correlation_analysis(
        ["2021-01", "2021-02", "2021-03", "2021-04"], [5.0, 5.0, 5.0, 5.0],
        ["2021-01", "2021-02", "2021-03", "2021-04"], [1.0, 2.0, 3.0, 4.0],
    )
    raw = json.dumps(result, default=str)
    reparsed = json.loads(raw)  # fails if the correlation was a bare NaN literal
    assert reparsed["pearson_correlation"] is None


def test_correlation_analysis_duplicate_periods_do_not_silently_explode():
    """A duplicated period on each side would cross-join into extra
    merged rows for what should be 3 real observations - shared_periods
    must reflect real distinct periods, not the cross-product (would be
    4 if the 2x2 duplicate on '2021-01' cross-joined)."""
    result = correlation_analysis(
        ["2021-01", "2021-01", "2021-02", "2021-03"], [1.0, 1.0, 2.0, 3.0],
        ["2021-01", "2021-01", "2021-02", "2021-03"], [10.0, 10.0, 20.0, 30.0],
    )
    assert result["shared_periods"] == 3


def test_causality_analysis_duplicate_periods_do_not_silently_explode():
    periods = [f"2021-{m:02d}" for m in range(1, 13)] * 1  # 12 unique
    dup_periods = periods[:6] + periods  # first 6 duplicated
    x_values = list(range(len(dup_periods)))
    y_values = list(range(len(dup_periods)))
    result = causality_analysis(dup_periods, x_values, dup_periods, y_values, max_lag=2)
    assert result["shared_periods"] == 12  # deduped, not the merge blow-up


def test_causality_analysis_result_is_json_safe():
    import random
    rng = random.Random(1)
    n = 20
    periods = [f"2021-{((m - 1) % 12) + 1:02d}" for m in range(1, n + 1)]
    x = [rng.uniform(-1, 1) for _ in range(n)]
    y = [rng.uniform(-1, 1) for _ in range(n)]
    result = causality_analysis(periods, x, y_periods=periods, y_values=y, max_lag=2)
    raw = json.dumps(result, default=str)
    json.loads(raw)


def test_threshold_filter_compare_inverted_range_no_crash():
    """low > high should just mean 'nothing is in window', not an error."""
    periods = [f"2021-{m:02d}" for m in range(1, 11)]
    result = threshold_filter_compare(periods, list(range(10)), periods, list(range(10, 20)), low=100, high=1)
    assert result["periods_in_window"] == 0


def test_threshold_filter_compare_result_is_json_safe():
    periods = [f"2021-{m:02d}" for m in range(1, 11)]
    result = threshold_filter_compare(periods, [5.0] * 10, periods, list(range(10)), low=0, high=10)
    json.dumps(result, default=str)


def test_threshold_filter_compare_rejects_non_numeric_low_high():
    """Bug found live: a string low/high reached pandas' .between() and
    raised a raw, unhelpful TypeError instead of a clean ValueError."""
    periods = [f"2021-{m:02d}" for m in range(1, 11)]
    with pytest.raises(ValueError):
        threshold_filter_compare(periods, list(range(10)), periods, list(range(10, 20)), low="abc", high="xyz")


# ---------------------------------------------------------------------
# Round 2 (2026-09-10): case-sensitivity guardrail, MAD=0 blind spot,
# change_detection crash on missing values - all confirmed live before
# being fixed here, not hypothetical.
# ---------------------------------------------------------------------

def test_lakehouse_query_allows_uppercase_table_name():
    """Bug found live: DuckDB is case-insensitive for unquoted
    identifiers (FROM GOLD_FACT_LONG runs fine against a table actually
    named gold_fact_long), but the known-table guardrail compared raw
    case and rejected it as 'unknown table' before DuckDB ever saw it."""
    result = lakehouse_query("SELECT COUNT(*) AS n FROM GOLD_FACT_LONG")
    assert result["row_count"] == 1


def test_lakehouse_query_allows_mixed_case_table_name():
    result = lakehouse_query("SELECT COUNT(*) AS n FROM Gold_Fact_Long")
    assert result["row_count"] == 1


def test_lakehouse_query_still_rejects_a_genuinely_unknown_table():
    """The case-insensitivity fix must not turn into 'accept anything' -
    a table that truly doesn't exist (in any case) must still be
    rejected."""
    with pytest.raises(ValueError):
        lakehouse_query("SELECT * FROM totally_made_up_table")


def test_anomaly_analysis_finds_an_outlier_when_mad_is_zero():
    """Bug found live: when >=half the series shares one value, MAD=0,
    and the old code returned all-zero scores unconditionally - a 100x
    outlier sitting right next to a flat run was reported as zero
    anomalies. This is a realistic shape for this project's own data
    (a policy rate held flat for months)."""
    periods = [f"2021-{m:02d}" for m in range(1, 11)]
    values = [5.0] * 9 + [500.0]
    result = anomaly_analysis(periods, values, threshold=3.0)
    assert len(result["anomalies"]) >= 1
    assert result["anomalies"][0]["period"] == "2021-10"


def test_anomaly_analysis_truly_constant_series_still_reports_nothing():
    """The MAD=0 fallback must not start hallucinating anomalies in a
    genuinely constant series (std is also 0 there - the fallback's own
    escape hatch)."""
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03"], [5.0, 5.0, 5.0])
    assert result["anomalies"] == []


def test_change_detection_none_value_is_skipped_not_a_crash():
    """Bug found live: a None in values (a real gap in the gold layer)
    reached a bare 'current - previous' subtraction and raised an
    unhandled TypeError instead of being skipped like a missing point
    is skipped everywhere else in this codebase."""
    result = change_detection(["2021-01", "2021-02", "2021-03"], [None, 5.0, 6.0])
    assert result["largest_changes"] == [{"period": "2021-03", "absolute_change": 1.0, "percent_change": 20.0}]


def test_change_detection_nan_value_is_skipped_not_a_crash():
    result = change_detection(["2021-01", "2021-02", "2021-03"], [float("nan"), 5.0, 6.0])
    assert result["largest_changes"] == [{"period": "2021-03", "absolute_change": 1.0, "percent_change": 20.0}]


def test_change_detection_from_zero_transition_is_reported_not_dropped():
    """Previously silently dropped entirely (can't divide by zero) - now
    reported with percent_change=None instead of vanishing, since a
    series starting from exactly zero is usually the most significant
    move, not the least."""
    result = change_detection(["2021-01", "2021-02", "2021-03"], [0.0, 5.0, 5.0])
    periods_reported = [c["period"] for c in result["largest_changes"]]
    assert "2021-02" in periods_reported
    entry = next(c for c in result["largest_changes"] if c["period"] == "2021-02")
    assert entry["absolute_change"] == 5.0
    assert entry["percent_change"] is None


def test_change_detection_from_zero_result_is_json_safe():
    result = change_detection(["2021-01", "2021-02"], [0.0, 5.0])
    raw = json.dumps(result, default=str)
    json.loads(raw)


# ---------------------------------------------------------------------
# Round 3 (2026-09-10): string-literal content leaking into the
# forbidden-token / semicolon / table-name checks, causing false
# positives on entirely harmless queries.
# ---------------------------------------------------------------------

def test_lakehouse_query_allows_forbidden_word_inside_a_string_literal():
    """Bug found live: 'please insert coin' as a literal value contains
    'insert ' as a raw substring, and the guardrail scanned the whole
    normalized SQL text for that substring regardless of whether it was
    inside a string - false-rejected as DML."""
    result = lakehouse_query("SELECT 'please insert coin' AS note FROM gold_fact_long LIMIT 1")
    assert result["rows"][0]["note"] == "please insert coin"


def test_lakehouse_query_allows_semicolon_inside_a_string_literal():
    """Bug found live: a literal 'a;b' tripped the 'no semicolon inside
    the query' check, which didn't know the ';' was inert string
    content, not a second statement."""
    result = lakehouse_query("SELECT 'a;b' AS x FROM gold_fact_long LIMIT 1")
    assert result["rows"][0]["x"] == "a;b"


def test_lakehouse_query_allows_from_clause_text_inside_a_string_literal():
    """Bug found live: a literal value containing the text 'from users'
    matched the FROM-name extraction regex and got rejected as
    referencing an unknown table, even though it was never a real
    FROM clause."""
    result = lakehouse_query("SELECT 'select * from users' AS note FROM gold_fact_long LIMIT 1")
    assert result["rows"][0]["note"] == "select * from users"


def test_lakehouse_query_still_blocks_real_stacked_statement_despite_literal_fix():
    """The string-literal fix must not accidentally widen the guardrail -
    a genuine second statement outside any string must still be blocked."""
    with pytest.raises(ValueError):
        lakehouse_query("SELECT 1; DROP TABLE gold_fact_long")


def test_lakehouse_query_still_blocks_dml_keyword_outside_a_literal():
    with pytest.raises(ValueError):
        lakehouse_query("UPDATE gold_fact_long SET Deger = 0")


def test_lakehouse_query_handles_escaped_quote_inside_literal():
    """SQL-standard escaped quote ('') must not prematurely end the
    literal and desynchronize the rest of the guardrail's parsing."""
    result = lakehouse_query("SELECT 'O''Brien; DROP TABLE x' AS name FROM gold_fact_long LIMIT 1")
    assert "O'Brien" in result["rows"][0]["name"]


# ---------------------------------------------------------------------
# _quarterly_sort_key: the non-zero-padded 'YYYY-Q' period bug
# ---------------------------------------------------------------------

def test_quarterly_sort_key_orders_q4_after_q3_within_a_year():
    """Bug found live: plain string comparison puts '2021-12' BEFORE
    '2021-9' ('1' < '9' at the first differing character), even though
    Q4 chronologically follows Q3. Confirmed this already affects
    api/gold_queries.py's get_regional_top(), which used SQL's
    MAX(Period) and got lucky only because the current data's most
    recent year has no Q4 yet."""
    periods = ["2021-3", "2021-6", "2021-9", "2021-12"]
    assert sorted(periods, key=_quarterly_sort_key) == ["2021-3", "2021-6", "2021-9", "2021-12"]
    assert sorted(periods) != ["2021-3", "2021-6", "2021-9", "2021-12"]  # proves plain string sort IS wrong here


def test_quarterly_sort_key_orders_across_years():
    periods = ["2022-3", "2021-12", "2021-3", "2022-12"]
    assert max(periods, key=_quarterly_sort_key) == "2022-12"


# ---------------------------------------------------------------------
# describe_series: edge cases
# ---------------------------------------------------------------------

def test_describe_series_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        describe_series(["2021-01", "2021-02"], [1.0])


def test_describe_series_rejects_empty_input():
    with pytest.raises(ValueError):
        describe_series([], [])


def test_describe_series_single_point_std_is_zero_not_nan():
    """pandas .std() on a single-element Series returns NaN (ddof=1
    needs >=2 points) - must be reported as 0.0, a valid JSON number,
    not silently produce a NaN that breaks json.dumps downstream."""
    result = describe_series(["2021-01"], [42.0])
    assert result["std"] == 0.0
    raw = json.dumps(result, default=str)
    json.loads(raw)


def test_describe_series_all_missing_values_no_crash():
    result = describe_series(["2021-01", "2021-02"], [None, float("nan")])
    assert result["missing_count"] == 2
    assert result["min"] is None
    assert result["first_period"] is None
    json.dumps(result, default=str)


def test_describe_series_first_value_zero_gives_none_percent_change_not_infinity():
    result = describe_series(["2021-01", "2021-02"], [0.0, 5.0])
    assert result["total_change"] == 5.0
    assert result["total_percent_change"] is None
    raw = json.dumps(result, default=str)
    json.loads(raw)  # would fail if percent_change were a bare Infinity token


def test_describe_series_result_is_always_json_safe():
    result = describe_series(["2021-01", "2021-02", "2021-03"], [1.0, None, 3.0])
    json.loads(json.dumps(result, default=str))


# ---------------------------------------------------------------------
# compare_regions: edge cases
# ---------------------------------------------------------------------

def test_compare_regions_rejects_zero_top_n():
    with pytest.raises(ValueError):
        compare_regions(top_n=0)


def test_compare_regions_rejects_negative_top_n():
    with pytest.raises(ValueError):
        compare_regions(top_n=-5)


def test_compare_regions_rejects_unknown_period():
    with pytest.raises(ValueError, match="Unknown period"):
        compare_regions(period="1999-3")


def test_compare_regions_rejects_unknown_compare_period():
    with pytest.raises(ValueError, match="Unknown compare_period"):
        compare_regions(compare_period="1999-3")


def test_compare_regions_rejects_compare_period_equal_to_period():
    with pytest.raises(ValueError, match="must differ"):
        compare_regions(period="2021-3", compare_period="2021-3")


def test_compare_regions_top_n_larger_than_available_regions_no_crash():
    """82 provinces exist - asking for far more than that must just
    return everything, not error."""
    result = compare_regions(top_n=10_000)
    assert len(result["ranking"]) <= 82


def test_compare_regions_result_is_json_safe():
    result = compare_regions(top_n=5, compare_period="2021-3")
    json.loads(json.dumps(result, default=str))


def test_compare_regions_latest_period_is_chronologically_correct():
    """Regression for the _quarterly_sort_key bug: the tool's own
    'default to latest' path must use the fixed comparator, not
    SQL MAX()/plain string ordering."""
    result = compare_regions(top_n=1)
    from src.agent.tools import _db
    con = _db()
    try:
        all_periods = [row[0] for row in con.execute("SELECT DISTINCT Period FROM gold_finturk_regional_quarterly").fetchall()]
    finally:
        con.close()
    assert result["period"] == max(all_periods, key=_quarterly_sort_key)
