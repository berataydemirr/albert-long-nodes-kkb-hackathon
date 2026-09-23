"""Tests for the shared read-only SELECT guardrail.

Factored out of tools.py's original lakehouse_query so query_imported_dataset
could reuse the exact same defenses rather than a second, possibly-drifting
copy - these tests exercise the guardrail directly, independent of which
database it happens to be protecting.
"""

import json

import duckdb
import pytest

from src.agent.sql_guard import MAX_ROWS, cte_names, referenced_tables, run_guarded_select


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE t (id INTEGER, name VARCHAR, amount DOUBLE)")
    c.execute("INSERT INTO t VALUES (1, 'a', 10.0), (2, 'b', NULL), (3, 'c', 30.0)")
    yield c
    c.close()


def test_plain_select_works(con):
    result = run_guarded_select(con, "SELECT * FROM t ORDER BY id")
    assert result["row_count"] == 3
    assert result["columns"] == ["id", "name", "amount"]


def test_extract_from_a_column_is_not_mistaken_for_a_table():
    """Live failure: EXTRACT(YEAR FROM Period_Date) was refused with
    "Unknown table(s) referenced: ['Period_Date']", and the model then
    wrongly told the user the Period_Date column did not exist."""
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE g AS SELECT DATE '2024-03-01' AS Period_Date, 5.0 AS Deger")
    try:
        result = run_guarded_select(c, "SELECT EXTRACT(YEAR FROM Period_Date) AS yil, AVG(Deger) FROM g GROUP BY 1")
    finally:
        c.close()
    assert result["rows"][0]["yil"] == 2024
    assert referenced_tables("SELECT extract ( month FROM d ) FROM g") == {"g"}


@pytest.mark.parametrize("sql", [
    # The subquery inside EXTRACT keeps its own FROM - still a table read.
    "SELECT EXTRACT(year FROM (SELECT * FROM read_csv('/etc/passwd')))",
    # An alias that happens to be an EXTRACT field name changes nothing.
    "SELECT 1 AS year FROM read_csv('/etc/passwd')",
])
def test_the_extract_fix_opens_no_path_to_reading_files(con, sql):
    with pytest.raises(ValueError, match="Unknown table"):
        run_guarded_select(con, sql)


def test_null_becomes_json_null_not_nan(con):
    """The exact bug found live in the two real callers of this function:
    astype(object) before .where(...) is required, or a float64 column's
    None is silently coerced back to np.nan and json.dumps emits an
    invalid bare NaN token."""
    result = run_guarded_select(con, "SELECT * FROM t WHERE id = 2")

    assert result["rows"][0]["amount"] is None
    serialized = json.dumps(result)
    assert "NaN" not in serialized
    assert json.loads(serialized) == result


@pytest.mark.parametrize("sql", [
    "SELECT * FROM t; DROP TABLE t;",
    "DROP TABLE t",
    "DELETE FROM t",
    "INSERT INTO t VALUES (4, 'x', 1.0)",
    "UPDATE t SET amount = 0",
    "ATTACH 'other.db'",
    "PRAGMA table_info(t)",
])
def test_unsafe_statements_are_rejected(con, sql):
    with pytest.raises(ValueError):
        run_guarded_select(con, sql)


def test_unknown_table_is_rejected(con):
    with pytest.raises(ValueError, match="Unknown table"):
        run_guarded_select(con, "SELECT * FROM some_other_table")


def test_cte_is_not_mistaken_for_an_unknown_table(con):
    result = run_guarded_select(con, "WITH x AS (SELECT * FROM t) SELECT * FROM x")
    assert result["row_count"] == 3


def test_string_literal_containing_a_forbidden_word_is_not_rejected(con):
    """A DML keyword inside a string LITERAL is data, not a statement -
    rejecting it would be a false positive on ordinary text."""
    con.execute("CREATE TABLE msgs (body VARCHAR)")
    result = run_guarded_select(con, "SELECT 'please insert coin' AS body")
    assert result["rows"][0]["body"] == "please insert coin"


def test_comment_hiding_a_second_statement_is_still_caught(con):
    with pytest.raises(ValueError):
        run_guarded_select(con, "SELECT * FROM t -- comment\n; DROP TABLE t;")


def test_row_cap_is_enforced(con):
    con.execute("CREATE TABLE big AS SELECT range AS n FROM range(1000)")
    result = run_guarded_select(con, "SELECT * FROM big", max_rows=50)
    assert result["row_count"] == 50
    assert result["truncated"] is True


def test_non_string_sql_is_rejected(con):
    with pytest.raises(ValueError, match="must be a string"):
        run_guarded_select(con, 12345)


def test_referenced_tables_finds_from_and_join():
    tables = referenced_tables("SELECT * FROM a JOIN b ON a.id = b.id")
    assert tables == {"a", "b"}


def test_cte_names_extracts_multiple_ctes():
    names = cte_names("WITH x AS (SELECT 1), y AS (SELECT 2) SELECT * FROM x, y")
    assert names == {"x", "y"}


def test_max_rows_constant_matches_the_documented_limit():
    assert MAX_ROWS == 200


def test_semicolon_inside_a_string_literal_is_not_a_multi_statement(con):
    result = run_guarded_select(con, "SELECT 'a;b;c' AS body")
    assert result["rows"][0]["body"] == "a;b;c"


def test_escaped_quote_inside_a_literal_does_not_break_blanking(con):
    """SQL escapes a literal quote as '' - the blanking regex must consume
    the whole literal in one match, not stop at the first quote."""
    result = run_guarded_select(con, "SELECT 'it''s fine, insert this' AS body")
    assert result["rows"][0]["body"] == "it's fine, insert this"


def test_from_keyword_inside_a_string_literal_is_not_a_table_reference(con):
    """'FROM' inside quoted text must not be parsed as a table name - it
    is data, not a FROM clause."""
    result = run_guarded_select(con, "SELECT 'far from perfect' AS body")
    assert result["rows"][0]["body"] == "far from perfect"
