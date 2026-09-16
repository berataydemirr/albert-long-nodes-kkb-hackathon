"""Adversarial / edge-case stress tests for src/agent/tools.py.

Not the happy-path tests in test_agent_tools.py - this file exists to
break things: SQL injection variants against lakehouse_query's
guardrail, zero-variance/NaN inputs, duplicate periods, malformed JSON
at the execute_tool boundary. Every failure found here was fixed in
tools.py, not worked around in the test.

Run: pytest tests/test_agent_tools_stress.py -v
"""
import json

import pytest

from src.agent.tools import (
    lakehouse_query, execute_tool,
    anomaly_analysis, change_detection, correlation_analysis,
    causality_analysis, threshold_filter_compare,
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
