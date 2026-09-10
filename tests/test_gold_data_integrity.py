"""Data-integrity tests against the REAL built gold database
(data/gold/lakehouse.duckdb), not synthetic fixtures. Fills a gap the
project's own README flags: "collectors, silver converters and
build_gold.py's own I/O aren't covered" - these tests exercise that I/O
by checking real, load-bearing invariants in its output.

Skips cleanly (not a failure) if the gold DB hasn't been built yet -
run `python -m src.gold.build_gold` first. CI runs the network-free unit
tests; this file is meant for local runs against a real build.

Run: pytest tests/test_gold_data_integrity.py -v
"""
import duckdb
import pandas as pd
import pytest

from config.settings import GOLD_DIR

DB_PATH = GOLD_DIR / "lakehouse.duckdb"

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="gold DB not built - run `python -m src.gold.build_gold` first")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(DB_PATH), read_only=True)
    yield c
    c.close()


# ---------------------------------------------------------------------
# Accounting identity: Assets == Liabilities, by definition of a
# balance sheet. Added alongside tablo_1.toplam_aktifler/toplam_pasifler
# specifically so this pipeline has a real cross-check beyond "did it
# run without crashing" - if a future change corrupts either series,
# this catches it even though nothing else would raise an error.
# ---------------------------------------------------------------------

def test_bddk_balance_sheet_assets_equal_liabilities(con):
    df = con.execute("""
        SELECT a.Period, a.Deger AS aktif, p.Deger AS pasif
        FROM gold_fact_long a
        JOIN gold_fact_long p ON a.Period = p.Period AND p.Seri_Kod = 'tablo_1.toplam_pasifler'
        WHERE a.Seri_Kod = 'tablo_1.toplam_aktifler'
    """).fetchdf()
    assert len(df) > 0, "expected tablo_1.toplam_aktifler/toplam_pasifler to exist - was the gold layer rebuilt after adding them?"
    diff_pct = ((df["aktif"] - df["pasif"]).abs() / df["aktif"]) * 100
    assert diff_pct.max() < 0.01, f"assets != liabilities by more than 0.01% in at least one period (max diff {diff_pct.max():.4f}%)"


def test_bddk_equity_never_exceeds_total_assets(con):
    """A sector-wide sanity bound: equity is a component of liabilities+
    equity, so it can never exceed total assets."""
    df = con.execute("""
        SELECT a.Period, a.Deger AS aktif, o.Deger AS ozkaynak
        FROM gold_fact_long a
        JOIN gold_fact_long o ON a.Period = o.Period AND o.Seri_Kod = 'tablo_1.toplam_ozkaynaklar'
        WHERE a.Seri_Kod = 'tablo_1.toplam_aktifler'
    """).fetchdf()
    assert len(df) > 0
    assert (df["ozkaynak"] < df["aktif"]).all()
    assert (df["ozkaynak"] > 0).all(), "sector-wide equity should never be reported as zero/negative"


def test_new_bddk_balance_sheet_series_have_full_monthly_coverage(con):
    """These 3 series were added from bronze/silver data that's already
    fully collected (1122/1122 files per README) - a partial/gappy
    result here would mean the pull itself is broken, not a real data
    gap."""
    for seri_kod in ("tablo_1.toplam_aktifler", "tablo_1.toplam_pasifler", "tablo_1.toplam_ozkaynaklar"):
        n = con.execute("SELECT COUNT(*) FROM gold_fact_long WHERE Seri_Kod = ?", [seri_kod]).fetchone()[0]
        assert n == 66, f"{seri_kod}: expected 66 months (2021-01..2026-06), got {n}"


# ---------------------------------------------------------------------
# tablo_6 (SME Loans): regression for the "wrong column picked"
# bug. build_bddk_monthly_item_facts()'s generic single-Value path
# (via load_table_long's 'last numeric column' fallback) silently
# picked a customer-COUNT column instead of the loan-amount column,
# reporting the SME total loan series as 0.0 TL for every recent month.
# Fixed with a bespoke loader (build_bddk_monthly_kobi_facts); these
# tests lock in that a future refactor can't quietly reintroduce it.
# ---------------------------------------------------------------------

def test_sme_loans_series_has_no_zero_values(con):
    df = con.execute("SELECT Period, Deger FROM gold_fact_long WHERE Seri_Kod = 'tablo_6.toplam_kobi_kredileri'").fetchdf()
    assert len(df) == 66
    zero_periods = df[df["Deger"] == 0]["Period"].tolist()
    assert not zero_periods, f"tablo_6.toplam_kobi_kredileri reported exactly 0 for: {zero_periods} - regression of the wrong-column bug"


def test_sme_loans_are_a_plausible_fraction_of_total_loans(con):
    """SME loans are a real subset of total sector loans - sanity bound,
    not an exact ratio claim (the two series come from different BDDK
    tables with slightly different scopes)."""
    df = con.execute("""
        SELECT s.Period, s.Deger AS sme, t.Deger AS total
        FROM gold_fact_long s
        JOIN gold_fact_long t ON s.Period = t.Period AND t.Seri_Kod = 'tablo_3.toplam_krediler'
        WHERE s.Seri_Kod = 'tablo_6.toplam_kobi_kredileri'
    """).fetchdf()
    assert len(df) > 0
    ratio = df["sme"] / df["total"]
    assert (ratio > 0).all() and (ratio < 1).all(), f"SME/total loan ratio out of [0,1]: min={ratio.min()}, max={ratio.max()}"


# ---------------------------------------------------------------------
# General gold_fact_long invariants - apply to every series, not just
# the newly added ones.
# ---------------------------------------------------------------------

def test_no_series_is_all_zero(con):
    """General safety net for the tablo_6 class of bug: a series that's
    0.0 in every single period almost certainly means the wrong source
    column was picked, not a real economic value (even NPL ratios,
    which can legitimately be small, aren't EXACTLY zero for 66
    straight months in real banking data)."""
    df = con.execute("""
        SELECT Seri_Kod, COUNT(*) AS n, SUM(CASE WHEN Deger = 0 THEN 1 ELSE 0 END) AS zero_n
        FROM gold_fact_long GROUP BY Seri_Kod
    """).fetchdf()
    all_zero = df[df["n"] == df["zero_n"]]
    assert all_zero.empty, f"series that are 0.0 in every period (likely wrong-column bug): {all_zero['Seri_Kod'].tolist()}"


def test_no_duplicate_period_seri_kod_pairs(con):
    """A duplicate (Period, Seri_Kod) row would silently double-count in
    any aggregation and inflate row counts without raising an error -
    the exact class of bug _align_series' drop_duplicates guards
    against on the agent-tools side (see tools.py); the gold layer
    itself must never produce one in the first place."""
    dupes = con.execute("""
        SELECT Seri_Kod, Period, COUNT(*) AS n
        FROM gold_fact_long
        GROUP BY Seri_Kod, Period
        HAVING COUNT(*) > 1
    """).fetchdf()
    assert dupes.empty, f"duplicate (Seri_Kod, Period) rows found:\n{dupes}"


def test_no_seri_kod_has_a_null_deger(con):
    """Birim (unit) can legitimately be NULL (unconfirmed unit, by
    design - see BDDK_FINTURK_TABLES's honesty rule), but Deger (the
    actual value) must never be null - a null value is a real gap that
    should have been excluded upstream, not silently carried into gold."""
    n = con.execute("SELECT COUNT(*) FROM gold_fact_long WHERE Deger IS NULL").fetchone()[0]
    assert n == 0


def test_every_seri_kod_has_a_non_empty_seri_adi(con):
    n = con.execute("SELECT COUNT(*) FROM gold_fact_long WHERE Seri_Adi IS NULL OR TRIM(Seri_Adi) = ''").fetchone()[0]
    assert n == 0


def test_monthly_periods_are_well_formed_yyyy_mm(con):
    """gold_fact_long is documented as the shared MONTHLY axis - every
    Period here must be 'YYYY-MM', not the non-zero-padded 'YYYY-Q'
    format the FinTurk regional table uses (that one has its own
    quarterly grain and its own table on purpose - see align.py)."""
    bad = con.execute(r"SELECT DISTINCT Period FROM gold_fact_long WHERE Period NOT SIMILAR TO '[0-9]{4}-[0-9]{2}'").fetchdf()
    assert bad.empty, f"malformed Period values in gold_fact_long: {bad['Period'].tolist()}"


def test_series_catalog_matches_actual_data_seri_kods(con):
    """lakehouse_schema()'s SERIES_CATALOG_QUERY (tools.py) - the exact
    query the agent uses to learn which Seri_Kod values it's allowed to
    reference - must not silently diverge from what Seri_Kod values
    actually exist in the fact table (e.g. a bad DISTINCT/JOIN dropping
    a series would mean the agent's own catalog lies to it about what
    it can query)."""
    from src.agent.tools import SERIES_CATALOG_QUERY
    catalog_kods = set(con.execute(SERIES_CATALOG_QUERY).fetchdf()["Seri_Kod"])
    actual_kods = set(con.execute("SELECT DISTINCT Seri_Kod FROM gold_fact_long").fetchdf()["Seri_Kod"])
    assert catalog_kods == actual_kods


# ---------------------------------------------------------------------
# gold_finturk_regional_quarterly: its own invariants (native quarterly
# grain, non-zero-padded periods - see _quarterly_sort_key in tools.py)
# ---------------------------------------------------------------------

def test_regional_quarterly_periods_are_well_formed(con):
    bad = con.execute(r"SELECT DISTINCT Period FROM gold_finturk_regional_quarterly WHERE Period NOT SIMILAR TO '[0-9]{4}-(3|6|9|12)'").fetchdf()
    assert bad.empty, f"malformed Period values in gold_finturk_regional_quarterly: {bad['Period'].tolist()}"


def test_regional_quarterly_has_82_provinces_every_quarter(con):
    """81 provinces + 'YURT DISI' (abroad) = 82, confirmed in the
    project's own docs - a quarter with a different count means a
    partial/corrupted pull for that quarter."""
    counts = con.execute("SELECT Period, COUNT(DISTINCT Bolge) AS n FROM gold_finturk_regional_quarterly GROUP BY Period").fetchdf()
    bad = counts[counts["n"] != 82]
    assert bad.empty, f"quarters with != 82 regions:\n{bad}"


def test_regional_npl_ratio_is_a_plausible_fraction(con):
    """NPL_Orani_Yaklasik = Takipteki_Alacak / Toplam_Nakdi_Kredi - a
    non-performing-loan ratio outside roughly [0, 1] would indicate a
    unit mismatch or a join gone wrong (e.g. cross-joined rows), not a
    real value seen in Turkish banking data."""
    bad = con.execute("SELECT * FROM gold_finturk_regional_quarterly WHERE NPL_Orani_Yaklasik < 0 OR NPL_Orani_Yaklasik > 1").fetchdf()
    assert bad.empty, f"implausible NPL ratios found:\n{bad}"
