"""Tests for promoting the full BDDK silver archive into gold.

WHY THIS EXISTS: a live question asked for TP/YP deposit maturity
structure (3 aya kadar vs 3 aydan fazla, 2021-01..2025-12) and the agent
answered "lakehouse'da yok" - while tablo_9 and tablo_10 carried exactly
those rows, with those maturity columns, for all 66 months, sitting in
data/silver. These tests assert against the REAL silver archive (it is
tracked in git, so CI has it too), not fixtures: the point is to prove
the specific data that question needed is now reachable, which a
synthetic fixture could not show.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from src.agent.tools import DB_PATH as TOOLS_DB_PATH, search_bddk_items
from src.gold import build_bddk_full as bf

TP_DEPOSIT_ROW = "TP Mevduat / Katılım Fonları - Yurt İçi Yerleşik"
FX_DEPOSIT_ROW = "Döviz Tevdiat Hesabı / Katılım Fonları - Yurt İçi Yerleşik"


# --- donemId arithmetic -------------------------------------------------
# The weekly bulletin addresses a period by an opaque counter. Every
# reference point below is one config/settings.py recorded from live
# reconnaissance; if the arithmetic drifts, every weekly date is silently
# wrong, so they are pinned here rather than trusted.

@pytest.mark.parametrize("donem_id,expected", [
    (367, date(2021, 1, 8)),
    (419, date(2022, 1, 7)),
    (471, date(2023, 1, 6)),
    (523, date(2024, 1, 5)),
    (575, date(2025, 1, 3)),
    (627, date(2026, 1, 2)),
    (652, date(2026, 6, 26)),
])
def test_donem_id_maps_to_the_documented_week(donem_id, expected):
    assert bf.donem_id_to_date(donem_id) == expected


# --- melt behaviour -----------------------------------------------------

def test_melt_drops_non_numeric_cells_instead_of_carrying_text():
    """The bulletin writes '-' for "not applicable". Carried through as
    text it would break every numeric comparison downstream, so those
    cells must disappear rather than become a string or a zero."""
    frame = pd.DataFrame({"Item": ["a", "b"], "TP": [5, "-"], "YP": ["", 7]})

    melted = bf._melt(frame, ("Item",))

    assert sorted(melted["Deger"].tolist()) == [5.0, 7.0]
    assert "-" not in melted["Deger"].astype(str).tolist()


def test_melt_returns_empty_when_there_is_nothing_but_bookkeeping_columns():
    frame = pd.DataFrame({"Item": ["a"], "Sequence": [1]})

    assert bf._melt(frame, ("Item", "Sequence")).empty


# --- monthly: the table the failing question actually needed ------------

@pytest.fixture(scope="module")
def con():
    """One build for the whole module, written into an in-memory DuckDB.

    Every table fixture below reads back OUT of DuckDB rather than using
    the builder's DataFrame directly, so the assertions cover what the
    agent will actually query - type coercion on write included - and the
    full archive is built once (~20s) instead of once per fixture.

    The sentinel table is created BEFORE the write so the additive-only
    guarantee can be checked afterwards."""
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE gold_fact_long AS SELECT 'x' AS Seri_Kod, 1 AS Deger")
    bf.write_bddk_full_tables(connection)
    return connection


@pytest.fixture(scope="module")
def monthly(con):
    return con.execute(f"SELECT * FROM {bf.MONTHLY_TABLE}").fetchdf()


def test_monthly_covers_every_bulletin_table_and_the_full_period_range(monthly):
    assert monthly["Tablo"].nunique() == 17
    assert monthly["Period"].min() == "2021-01"
    assert monthly["Period"].max() >= "2026-06"
    # ~132k cells were counted on disk; a big drop means files stopped
    # being read, which no shape assertion above would catch.
    assert len(monthly) > 100_000


def test_monthly_exposes_the_tp_and_fx_deposit_rows_the_live_question_needed(monthly):
    tablo_9 = monthly[monthly["Tablo"] == "tablo_9"]
    rows = set(tablo_9["Satir"])

    assert TP_DEPOSIT_ROW in rows
    assert FX_DEPOSIT_ROW in rows


def test_monthly_exposes_the_maturity_columns_for_the_deposit_split(monthly):
    """"3 aya kadar" / "3 aydan fazla" is not a stored column - it is a
    SUM over these buckets, so all six must survive the melt."""
    tablo_10 = monthly[monthly["Tablo"] == "tablo_10"]
    kolonlar = set(tablo_10["Kolon"])

    assert {"Vadesiz", "1 Aya Kadar", "1-3 Ay Arası",
            "3-6 Ay Arası", "6-12 Ay Arası", "1  Yıl Üzeri"} <= kolonlar


def test_the_deposit_maturity_question_is_now_answerable_end_to_end(monthly):
    """The actual query the failing turn needed: TP deposits split into
    up-to-3-months vs over-3-months, monthly, for a real period."""
    row = monthly[
        (monthly["Tablo"] == "tablo_10")
        & (monthly["Satir"] == TP_DEPOSIT_ROW)
        & (monthly["Period"] == "2021-01")
    ]
    short_end = row[row["Kolon"].isin(["Vadesiz", "1 Aya Kadar", "1-3 Ay Arası"])]["Deger"].sum()
    long_end = row[row["Kolon"].isin(["3-6 Ay Arası", "6-12 Ay Arası", "1  Yıl Üzeri"])]["Deger"].sum()
    total = row[row["Kolon"] == "Toplam"]["Deger"].iloc[0]

    assert short_end > 0 and long_end > 0
    # The buckets are a partition of the row, so they must reconstruct it.
    assert short_end + long_end == pytest.approx(total, rel=0.001)


def test_monthly_values_are_not_stock_snapshots_of_one_month_only(monthly):
    """Guards the "sadece o aydaki veri, stok olmasın" requirement in the
    question: consecutive months must carry genuinely different values,
    not a single repeated snapshot."""
    series = monthly[
        (monthly["Tablo"] == "tablo_10")
        & (monthly["Satir"] == TP_DEPOSIT_ROW)
        & (monthly["Kolon"] == "Toplam")
    ].sort_values("Period")

    assert len(series) >= 60
    assert series["Deger"].nunique() > 50


def test_cumulative_income_statement_rows_are_flagged(monthly):
    """Every tablo_2 (Kar Zarar) row is a year-to-date cumulative - a
    monthly figure there is a DIFFERENCE between months, not the stored
    value. Shipping it unflagged is how an agent silently reports a
    12-month total as if it were December's own profit."""
    tablo_2 = monthly[monthly["Tablo"] == "tablo_2"]

    assert tablo_2["Kumulatif"].any()
    # ...and a plain stock table must not be flagged the same way.
    tablo_9 = monthly[monthly["Tablo"] == "tablo_9"]
    assert not tablo_9["Kumulatif"].fillna(False).all()


def test_row_hierarchy_is_preserved_so_totals_are_not_double_counted(monthly):
    """The bulletin marks a top-level item bold and its breakdown italic.
    Without that marker a SUM() over rows adds parents to their own
    children."""
    tablo_9 = monthly[monthly["Tablo"] == "tablo_9"]
    seviye = set(tablo_9["Seviye"].dropna())

    assert "bold" in seviye and "italic" in seviye
    parent = tablo_9[(tablo_9["Satir"] == TP_DEPOSIT_ROW) & (tablo_9["Kolon"] == "Toplam")]
    assert (parent["Seviye"] == "bold").all()


def test_monthly_carries_each_table_s_own_unit(monthly):
    """Units differ per table (tablo_5 is 'bin TL' where most are 'milyon
    TL'), so one shared unit would misstate whole tables by 1000x."""
    units = monthly.groupby("Tablo")["Birim"].first()

    assert units["tablo_1"] == "milyon TL"
    assert units["tablo_5"] == "bin TL"


# --- weekly -------------------------------------------------------------

@pytest.fixture(scope="module")
def weekly(con):
    return con.execute(f"SELECT * FROM {bf.WEEKLY_TABLE}").fetchdf()


def test_weekly_covers_every_table_and_both_time_axes(weekly):
    assert weekly["Tablo"].nunique() == 9
    assert weekly["Kolon"].isin(["TP", "YP", "TOPLAM"]).all()
    # Period exists so weekly can be aligned with monthly gold tables,
    # Tarih so the actual week is never lost to that rounding.
    assert weekly["Period"].min() == "2021-01"
    assert weekly.groupby("Period")["Tarih"].nunique().max() >= 4


def test_weekly_is_big_enough_to_be_the_whole_archive(weekly):
    """2,574 source CSVs were collected; one series used to reach gold."""
    assert len(weekly) > 100_000


# --- finturk ------------------------------------------------------------

@pytest.fixture(scope="module")
def finturk(con):
    return con.execute(f"SELECT * FROM {bf.FINTURK_TABLE}").fetchdf()


def test_finturk_covers_every_table_and_all_provinces(finturk):
    assert finturk["Tablo"].nunique() == 7
    # 81 provinces plus an "abroad" bucket.
    assert finturk["Bolge"].nunique() >= 81


def test_finturk_exposes_metrics_the_curated_regional_table_never_had(finturk):
    """gold_regional_long carries 35 curated metrics; tablo_7's gold
    deposits and tablo_4's sectoral loan columns are not among them."""
    kolonlar = set(finturk["Kolon"])

    assert "Toplam Altın Mevduatı" in kolonlar
    assert any("Tekstil" in k for k in kolonlar)


# --- writing ------------------------------------------------------------

def test_write_creates_the_three_tables_and_touches_nothing_else(con):
    """The hard constraint on this work: it is additive. The sentinel
    table the fixture created before the build must survive untouched."""
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}

    assert {bf.MONTHLY_TABLE, bf.WEEKLY_TABLE, bf.FINTURK_TABLE} <= tables
    assert con.execute("SELECT * FROM gold_fact_long").fetchall() == [("x", 1)]


def test_written_tables_answer_the_deposit_question_in_plain_sql(con):
    """End to end through DuckDB, the way the agent will actually reach
    it: one grouped SELECT, no per-series lookup."""
    rows = con.execute(f"""
        SELECT Period,
               SUM(CASE WHEN Kolon IN ('Vadesiz','1 Aya Kadar','1-3 Ay Arası')
                        THEN Deger END) AS uc_aya_kadar,
               SUM(CASE WHEN Kolon IN ('3-6 Ay Arası','6-12 Ay Arası','1  Yıl Üzeri')
                        THEN Deger END) AS uc_aydan_fazla
        FROM {bf.MONTHLY_TABLE}
        WHERE Tablo = 'tablo_10' AND Satir = ?
          AND Period BETWEEN '2021-01' AND '2025-12'
        GROUP BY Period ORDER BY Period
    """, [TP_DEPOSIT_ROW]).fetchall()

    assert len(rows) == 60  # every month of 2021-01..2025-12, none missing
    assert all(short > 0 and long > 0 for _, short, long in rows)


# --- the discovery tool -------------------------------------------------
# These hit the REAL lakehouse (search_bddk_items opens it through
# tools._db), so they skip when it has not been built, same as the rest of
# the suite's DB-backed tests.

needs_db = pytest.mark.skipif(
    not TOOLS_DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


@needs_db
def test_search_finds_the_deposit_row_from_a_lowercase_turkish_keyword():
    """The labels are title-case Turkish with İ/ı; a keyword typed in
    lowercase must still match or the tool is useless in practice."""
    result = search_bddk_items(query="mevduat vade", kaynak="aylik")

    assert result["row_count"] > 0
    labels = [r["filtre"].get("Satir") for r in result["results"]]
    assert any(TP_DEPOSIT_ROW == label for label in labels)


@needs_db
def test_search_returns_filters_that_can_be_used_verbatim_in_sql():
    """The whole point of the tool: what it returns must drop straight
    into a WHERE clause and select real rows."""
    result = search_bddk_items(query="döviz tevdiat", kaynak="aylik")
    match = next(r for r in result["results"] if r["filtre"].get("Satir") == FX_DEPOSIT_ROW)

    con = duckdb.connect(str(TOOLS_DB_PATH), read_only=True)
    try:
        count = con.execute(
            f"SELECT count(*) FROM {bf.MONTHLY_TABLE} WHERE Tablo = ? AND Satir = ?",
            [match["filtre"]["Tablo"], match["filtre"]["Satir"]],
        ).fetchone()[0]
    finally:
        con.close()

    assert count > 0
    assert "Toplam" in match["kolonlar"]


@needs_db
def test_search_can_be_narrowed_to_one_bulletin():
    weekly_only = search_bddk_items(query="kredi", kaynak="haftalik")

    assert weekly_only["searched_tables"] == [bf.WEEKLY_TABLE]
    assert {r["kaynak"] for r in weekly_only["results"]} == {"haftalik"}


@needs_db
def test_search_with_no_match_explains_what_to_do_instead_of_returning_nothing():
    """A bare empty result reads to the model as "this data does not
    exist" - the failure mode this whole feature exists to fix."""
    result = search_bddk_items(query="zzzz-boyle-bir-kalem-yok")

    assert result["row_count"] == 0
    assert result["results"] == []
    assert "next_step" in result and result["next_step"]
