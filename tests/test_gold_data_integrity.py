"""Regression tests for the gold layer's structural guarantees.

WHY THIS FILE EXISTS: every check here corresponds to a defect that was
found live in the built lakehouse and that produced a WRONG ANSWER
SILENTLY - no exception, no empty result, just fewer rows or the wrong
rows. Those are the worst kind for an LLM-facing database, because the
model writes correct SQL, gets plausible-looking output, and reasons on
top of it.

The defects:
  1. gold_regional_long / gold_finturk_regional_quarterly wrote Period as
     '2026-6' while every monthly table wrote '2026-06'. A JOIN on Period
     matched only the December quarters (5 of 22); Q1/Q2/Q3 silently
     disappeared from every cross-table query.
  2. Because of the same padding, string ordering disagreed with calendar
     ordering inside a year ('2025-9' sorted above '2025-12'), so
     ORDER BY Period DESC LIMIT 4 returned the wrong quarters - and "son 4
     çeyrek" is a real question the mentors ask.
  3. Period was text-only, so no date arithmetic was possible at all.
  4. Nothing enforced key uniqueness, so a rebuild could duplicate rows
     without anything noticing.
"""

import duckdb
import pandas as pd
import pytest

from src.agent.regional import _normalize_period, _quarter_sort_key, _resolve_period
from src.gold.build_gold import (
    GOLD_PERIOD_TABLES,
    add_period_date_columns,
    assert_gold_integrity,
    create_gold_indexes,
)
from src.agent.series_resolver import DB_PATH

needs_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


def _memory_db_with(fact_rows: list[dict], regional_rows: list[dict]) -> duckdb.DuckDBPyConnection:
    """A throwaway in-memory lakehouse carrying just the two tables the
    integrity checks care about."""
    con = duckdb.connect(":memory:")
    con.register("fact_df", pd.DataFrame(fact_rows))
    con.execute("CREATE TABLE gold_fact_long AS SELECT * FROM fact_df")
    con.register("regional_df", pd.DataFrame(regional_rows))
    con.execute("CREATE TABLE gold_regional_long AS SELECT * FROM regional_df")
    return con


def _fact(period, seri="S1", bolge="TR", deger=1.0):
    return {"Period": period, "Bolge": bolge, "Seri_Kod": seri, "Deger": deger}


def _regional(period, bolge="ANKARA", metrik="kredi", deger=1.0):
    return {"Period": period, "Bolge": bolge, "Metrik": metrik, "Deger": deger}


# --- Period_Date ----------------------------------------------------------

def test_period_date_is_built_for_monthly_periods():
    con = _memory_db_with([_fact("2026-06"), _fact("2025-12", seri="S2")], [_regional("2026-06")])

    add_period_date_columns(con)

    rows = con.execute("SELECT Period, Period_Date FROM gold_fact_long ORDER BY Period").fetchall()
    assert [str(r[1]) for r in rows] == ["2025-12-01", "2026-06-01"]


def test_period_date_handles_annual_periods_without_a_month():
    """TUIK's life-expectancy rows carry '2021', not '2021-01'. Those must
    resolve to a real date too, not NULL."""
    con = _memory_db_with([_fact("2021"), _fact("2024", seri="S2")], [_regional("2026-06")])

    add_period_date_columns(con)

    rows = con.execute("SELECT Period, Period_Date FROM gold_fact_long ORDER BY Period").fetchall()
    assert [str(r[1]) for r in rows] == ["2021-01-01", "2024-01-01"]


def test_period_date_is_null_rather_than_raising_on_a_junk_period():
    """One malformed row must not take the whole build down - it is
    reported by assert_gold_integrity instead."""
    con = _memory_db_with([_fact("bozuk-donem")], [_regional("2026-06")])

    add_period_date_columns(con)

    assert con.execute("SELECT Period_Date FROM gold_fact_long").fetchone()[0] is None


def test_period_date_makes_chronological_ordering_correct():
    """The original bug in one assertion: as text '2025-9' sorts above
    '2025-12'; as a date it does not."""
    con = _memory_db_with(
        [_fact("2025-09"), _fact("2025-12", seri="S2")],
        [_regional("2025-09"), _regional("2025-12", metrik="npl")],
    )
    add_period_date_columns(con)

    latest = con.execute(
        "SELECT Period FROM gold_regional_long ORDER BY Period_Date DESC LIMIT 1"
    ).fetchone()[0]
    assert latest == "2025-12"


# --- Unique indexes -------------------------------------------------------

def test_unique_index_is_created_on_the_natural_key():
    con = _memory_db_with([_fact("2026-06")], [_regional("2026-06")])
    add_period_date_columns(con)

    create_gold_indexes(con)

    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_gold_fact_long_key" in indexes
    assert "idx_gold_regional_long_key" in indexes


def test_duplicate_rows_fail_the_build_with_the_offending_key():
    """Silently duplicated rows double every SUM the agent computes, so
    this must stop the build rather than be discovered in an answer."""
    con = _memory_db_with(
        [_fact("2026-06"), _fact("2026-06")],  # same (Period, Bolge, Seri_Kod)
        [_regional("2026-06")],
    )
    add_period_date_columns(con)

    with pytest.raises(ValueError, match="duplicate rows"):
        create_gold_indexes(con)


# --- assert_gold_integrity -----------------------------------------------

def test_integrity_passes_on_a_consistent_lakehouse():
    con = _memory_db_with(
        [_fact("2026-06"), _fact("2026-03", seri="S2")],
        [_regional("2026-06"), _regional("2026-03", metrik="npl")],
    )
    add_period_date_columns(con)

    assert_gold_integrity(con)  # must not raise


def test_integrity_rejects_the_unpadded_period_spelling():
    """The exact original defect: regional writing '2026-6'."""
    con = _memory_db_with([_fact("2026-06")], [_regional("2026-6")])
    add_period_date_columns(con)

    with pytest.raises(ValueError) as exc:
        assert_gold_integrity(con)
    assert "YYYY-MM" in str(exc.value)


def test_integrity_rejects_regional_periods_that_cannot_join_the_fact_table():
    """A regional quarter with no matching national Period silently
    disappears from any JOIN, which is how the original bug hid."""
    con = _memory_db_with([_fact("2026-06")], [_regional("2026-06"), _regional("2025-03")])
    add_period_date_columns(con)

    with pytest.raises(ValueError, match="silently drop"):
        assert_gold_integrity(con)


def test_integrity_rejects_text_ordering_that_disagrees_with_the_calendar():
    con = _memory_db_with(
        [_fact("2025-9"), _fact("2025-12", seri="S2")],
        [_regional("2025-9"), _regional("2025-12", metrik="npl")],
    )
    add_period_date_columns(con)

    with pytest.raises(ValueError) as exc:
        assert_gold_integrity(con)
    # It trips the format check first, which is the more specific message.
    assert "YYYY-MM" in str(exc.value) or "wrong rows" in str(exc.value)


# --- Period spelling tolerance in the regional tool ----------------------

@pytest.mark.parametrize("spelling,expected", [
    ("2026-6", "2026-06"),
    ("2026-06", "2026-06"),
    (" 2026-6 ", "2026-06"),
    ("2021-3", "2021-03"),
    ("2021-12", "2021-12"),
])
def test_normalize_period_accepts_both_spellings(spelling, expected):
    assert _normalize_period(spelling) == expected


def test_normalize_period_leaves_junk_untouched_for_the_caller_to_reject():
    assert _normalize_period("bozuk") == "bozuk"


def test_resolve_period_finds_a_padded_quarter_from_the_old_unpadded_request():
    """Earlier chat turns, the old tool schema and hand-typed questions all
    still carry '2026-6'. After the gold fix the table stores '2026-06',
    and that must not turn into 'Unknown period' mid-demo."""
    assert _resolve_period("2026-6", ["2026-03", "2026-06"]) == "2026-06"


def test_resolve_period_still_rejects_a_quarter_that_really_is_missing():
    with pytest.raises(ValueError, match="Unknown period"):
        _resolve_period("2019-03", ["2026-03", "2026-06"])


def test_quarter_sort_key_still_handles_both_spellings():
    """Kept deliberately format-agnostic so this module stays correct
    against a database built before the padding fix."""
    mixed = ["2025-9", "2025-12", "2026-03", "2025-06"]
    assert sorted(mixed, key=_quarter_sort_key) == ["2025-06", "2025-9", "2025-12", "2026-03"]


# --- The real, built lakehouse -------------------------------------------

@needs_db
def test_built_lakehouse_uses_one_period_spelling_everywhere():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for table in GOLD_PERIOD_TABLES:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()[0]
            if not exists:
                continue
            bad = con.execute(
                f"""SELECT DISTINCT Period FROM "{table}"
                    WHERE NOT regexp_matches(Period, '^[0-9]{{4}}(-[0-9]{{2}})?$') LIMIT 5"""
            ).fetchall()
            assert not bad, f"{table} has non-'YYYY-MM' periods: {[b[0] for b in bad]}"
    finally:
        con.close()


@needs_db
def test_built_lakehouse_regional_periods_all_join_the_fact_table():
    """The headline regression: 17 of 22 regional quarters used to vanish
    from any JOIN on Period."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        unmatched = con.execute(
            """SELECT DISTINCT r.Period FROM gold_regional_long r
               WHERE r.Period NOT IN (SELECT DISTINCT Period FROM gold_fact_long)"""
        ).fetchall()
        assert not unmatched, f"regional periods with no national match: {[u[0] for u in unmatched]}"
    finally:
        con.close()


@needs_db
def test_built_lakehouse_orders_quarters_chronologically_as_plain_text():
    """'ORDER BY Period DESC LIMIT 4' is what the model actually writes;
    it has to return the real last four quarters."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        by_text = [r[0] for r in con.execute(
            "SELECT DISTINCT Period FROM gold_regional_long ORDER BY Period DESC LIMIT 4"
        ).fetchall()]
        by_date = [r[0] for r in con.execute(
            "SELECT DISTINCT Period, Period_Date FROM gold_regional_long ORDER BY Period_Date DESC LIMIT 4"
        ).fetchall()]
        assert by_text == by_date, f"text order {by_text} != chronological order {by_date}"
    finally:
        con.close()


@needs_db
def test_built_lakehouse_has_period_date_on_every_gold_table():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for table in GOLD_PERIOD_TABLES:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()[0]
            if not exists:
                continue
            columns = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
            assert "Period_Date" in columns, f"{table} is missing Period_Date"
            nulls = con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE Period_Date IS NULL').fetchone()[0]
            assert nulls == 0, f"{table} has {nulls} row(s) with an unparseable Period"
    finally:
        con.close()


@needs_db
def test_built_lakehouse_has_no_duplicate_natural_keys():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for table, key in GOLD_PERIOD_TABLES.items():
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()[0]
            if not exists:
                continue
            columns = ", ".join(f'"{c}"' for c in key)
            dupes = con.execute(
                f'SELECT {columns} FROM "{table}" GROUP BY {columns} HAVING COUNT(*) > 1 LIMIT 3'
            ).fetchall()
            assert not dupes, f"{table} has duplicate {key}: {dupes}"
    finally:
        con.close()
