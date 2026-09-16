"""Tests for PDF table extraction/import (src/agent/pdf_import.py).

Real PDFs are built here with reportlab (test-only dependency) rather
than hand-crafted bytes or a mocked pdfplumber - pdfplumber's table
detection depends on actual vector line graphics on the page, so a fake
that skips PDF rendering entirely would not catch a real regression in
how pdfplumber sees the file.
"""
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from src.agent.pdf_import import import_pdf_table, list_pdf_tables


@pytest.fixture(autouse=True)
def _fresh_lakehouse_db(monkeypatch, tmp_path):
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "lakehouse_test.duckdb")
    yield


def _make_pdf(path, rows, page_break_before=None, repeat_header=False):
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    elements = []
    if page_break_before:
        from reportlab.platypus import PageBreak, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        elements.append(Paragraph("Not a table - just text.", getSampleStyleSheet()["Normal"]))
        elements.append(PageBreak())
    # repeatRows=1 makes reportlab re-draw the header row on every page a
    # long table spills onto - without it, a table split across pages has
    # a header ONLY on the first page, which is not representative of the
    # real-world multi-page tables (BDDK bulletins, bank statements) this
    # is meant to simulate.
    t = Table(rows, repeatRows=1 if repeat_header else 0)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)
    doc.build(elements)


@pytest.fixture
def simple_pdf(tmp_path):
    p = tmp_path / "hastane.pdf"
    _make_pdf(p, [
        ["Il", "Hastane_Sayisi", "Yatak_Sayisi"],
        ["Ankara", "42", "15200"],
        ["Istanbul", "88", "31500"],
        ["Izmir", "35", "11400"],
    ])
    return p


def test_list_pdf_tables_finds_the_table(simple_pdf):
    result = list_pdf_tables(simple_pdf)
    assert result["page_count"] == 1
    assert result["tables"] == [{"page": 0, "table_index": 0, "row_count": 4, "column_count": 3}]


def test_import_pdf_table_happy_path(simple_pdf):
    result = import_pdf_table(simple_pdf, "hastane_pdf")

    assert result["dataset_name"] == "hastane_pdf"
    assert result["row_count"] == 3
    assert result["column_count"] == 3
    types = {c["column"]: c["inferred_type"] for c in result["columns"]}
    assert types["Hastane_Sayisi"] == "numeric"  # numeric-looking PDF text is converted, not left as text
    assert types["Il"] == "text"


def test_import_pdf_table_page_with_no_table_is_a_page_containing_only_text(tmp_path):
    p = tmp_path / "text_then_table.pdf"
    _make_pdf(p, [["A", "B"], ["1", "2"]], page_break_before=True)

    # page 0 is the text-only page (no table); page 1 has the real table.
    with pytest.raises(ValueError, match="No table found on page 0"):
        import_pdf_table(p, "should_fail", page=0)

    result = import_pdf_table(p, "from_page_1", page=1)
    assert result["row_count"] == 1


def test_import_pdf_table_page_out_of_range(simple_pdf):
    with pytest.raises(ValueError, match="out of range"):
        import_pdf_table(simple_pdf, "x", page=5)


def test_import_pdf_table_table_index_out_of_range(simple_pdf):
    with pytest.raises(ValueError, match="out of range"):
        import_pdf_table(simple_pdf, "x", table_index=3)


def test_import_pdf_table_rejects_duplicate_header(tmp_path):
    p = tmp_path / "dup.pdf"
    _make_pdf(p, [["Deger", "Deger"], ["1", "2"]])

    with pytest.raises(ValueError, match="Duplicate column"):
        import_pdf_table(p, "dup_pdf")


def test_import_pdf_table_rejects_blank_header_cell(tmp_path):
    p = tmp_path / "blank_header.pdf"
    _make_pdf(p, [["Il", ""], ["Ankara", "1"]])

    with pytest.raises(ValueError, match="blank column header"):
        import_pdf_table(p, "blank_header_pdf")


def test_import_pdf_table_flattens_a_merged_header(tmp_path):
    """4. Birlestirilmis Excel basliklari - the PDF equivalent, found live
    against a real Hazine ve Maliye Bakanligi budget table: a year merged
    over quarter columns leaves pdfplumber's raw grid with the year on
    the merge's first cell and blank on the rest, with the real
    sub-header (the quarter names) in the row right after. Must recover
    real column names ('2025_Ç1' etc.), not reject the whole table the
    way an unrecoverable blank header does."""
    p = tmp_path / "merged_header.pdf"
    _make_pdf(p, [
        ["Il", "2025", "", ""],
        ["", "Ç1", "Ç2", "Ç3"],
        ["Ankara", "100", "110", "120"],
        ["Istanbul", "200", "210", "220"],
    ])

    result = import_pdf_table(p, "merged_header_pdf")
    assert result["row_count"] == 2  # the sub-header row is not counted as data
    columns = [c["column"] for c in result["columns"]]
    assert columns == ["Il", "2025_Ç1", "2025_Ç2", "2025_Ç3"]
    assert result["data_quality"]["merged_header_flattened"] is True


def test_import_pdf_table_still_rejects_a_header_that_stays_blank_after_flattening(tmp_path):
    """A genuinely broken header (not a resolvable merge - here, two
    identical sub-header labels under two different blank top cells)
    must still be rejected, not guessed at further, even once
    merged-header flattening exists as a first attempt: 'Il_A' would
    collide with itself, so the collision check that already guards
    _flatten_merged_excel_header's Excel counterpart catches this too."""
    p = tmp_path / "still_blank.pdf"
    _make_pdf(p, [
        ["Il", "", ""],
        ["", "A", "A"],
        ["Ankara", "1", "2"],
        ["Istanbul", "3", "4"],
    ])

    with pytest.raises(ValueError, match="blank column header|Duplicate column"):
        import_pdf_table(p, "still_blank_pdf")


def test_import_pdf_table_rejects_header_only_table(tmp_path):
    p = tmp_path / "header_only.pdf"
    _make_pdf(p, [["Il", "Deger"]])

    with pytest.raises(ValueError, match="no data rows"):
        import_pdf_table(p, "header_only_pdf")


def test_import_pdf_table_converts_turkish_formatted_numbers(tmp_path):
    """A PDF has no cell dtype - '1.234,56' (Turkish thousands/decimal
    separators) must be recognized as 1234.56, not left as unusable text
    or misparsed as 1.23456."""
    p = tmp_path / "turkish_numbers.pdf"
    _make_pdf(p, [
        ["Il", "Tutar"],
        ["Ankara", "1.234,56"],
        ["Istanbul", "2.500,00"],
        ["Izmir", "999,00"],
    ])

    result = import_pdf_table(p, "turkish_numbers_pdf")
    types = {c["column"]: c["inferred_type"] for c in result["columns"]}
    assert types["Tutar"] == "numeric"

    from src.agent.data_import import query_imported_dataset
    rows = query_imported_dataset("turkish_numbers_pdf", "SELECT Tutar FROM turkish_numbers_pdf ORDER BY Tutar")["rows"]
    assert [r["Tutar"] for r in rows] == [999.0, 1234.56, 2500.0]


def test_import_pdf_table_does_not_corrupt_international_decimal_notation(tmp_path):
    """Real bug found live: the original normalization unconditionally
    stripped every '.' as if it were always a Turkish thousands
    separator, so a plain international-notation value like '2.009'
    silently became 2009 - a 1000x corruption, no error raised. A
    period-only value (no comma anywhere) must be read as an ordinary
    decimal point instead."""
    p = tmp_path / "international_numbers.pdf"
    _make_pdf(p, [
        ["Il", "Oran"],
        ["Ankara", "2.009"],
        ["Istanbul", "8.176"],
        ["Izmir", "5.500"],
    ])

    result = import_pdf_table(p, "intl_numbers_pdf")
    types = {c["column"]: c["inferred_type"] for c in result["columns"]}
    assert types["Oran"] == "numeric"

    from src.agent.data_import import query_imported_dataset
    rows = query_imported_dataset("intl_numbers_pdf", "SELECT Oran FROM intl_numbers_pdf ORDER BY Oran")["rows"]
    assert [r["Oran"] for r in rows] == [2.009, 5.5, 8.176]


def test_import_pdf_table_still_handles_turkish_numbers_mixed_with_plain_integers(tmp_path):
    """A column mixing a Turkish-decimal value with a plain integer (no
    separator at all) - both must parse correctly, independently, since
    the normalization now runs per-cell rather than per-column."""
    p = tmp_path / "mixed_numbers.pdf"
    _make_pdf(p, [
        ["Il", "Deger"],
        ["Ankara", "1.234,56"],
        ["Istanbul", "42"],
    ])

    from src.agent.data_import import query_imported_dataset
    import_pdf_table(p, "mixed_numbers_pdf")
    rows = query_imported_dataset("mixed_numbers_pdf", "SELECT Deger FROM mixed_numbers_pdf ORDER BY Deger")["rows"]
    assert [r["Deger"] for r in rows] == [42.0, 1234.56]


def test_import_pdf_table_stitches_a_continuation_page_by_default(tmp_path):
    """6. Sayfalara bölünmüş PDF tabloları: a table split across two pages
    (same header repeated, e.g. via reportlab's repeatRows, or a real
    multi-page kâr raporu) must be combined into ONE import by default -
    not silently truncated to whichever page was requested. Found live
    before stitching existed: importing page 0 alone silently dropped
    every row that had landed on page 1."""
    p = tmp_path / "continued_table.pdf"
    header = ["Il", "Deger"]
    all_rows = [["Il_%d" % i, str(i)] for i in range(40)]  # enough rows to force a page break
    _make_pdf(p, [header] + all_rows, repeat_header=True)

    preview = list_pdf_tables(p)
    assert preview["page_count"] >= 2, "test fixture did not actually span multiple pages"

    result = import_pdf_table(p, "continued_pdf", page=0)
    assert result["continuation_pages"] == [1]
    assert result["stitched"] is True
    assert result["row_count"] == 40  # every row recovered, not just page 0's share
    assert "stitched into this import" in result["note"]

    from src.agent.data_import import query_imported_dataset
    rows = query_imported_dataset("continued_pdf", "SELECT COUNT(*) AS n FROM continued_pdf")["rows"]
    assert rows[0]["n"] == 40


def test_import_pdf_table_stitching_can_be_disabled(tmp_path):
    """A caller that knows two pages sharing a header are actually
    unrelated tables can opt out and get only the requested page, the
    pre-stitching behavior."""
    p = tmp_path / "continued_table2.pdf"
    header = ["Il", "Deger"]
    all_rows = [["Il_%d" % i, str(i)] for i in range(40)]
    _make_pdf(p, [header] + all_rows, repeat_header=True)

    result = import_pdf_table(p, "unstitched_pdf", page=0, stitch_continuation_pages=False)
    assert result["continuation_pages"] == [1]
    assert result["stitched"] is False
    assert result["row_count"] < 40
    assert "NOT stitched in" in result["note"]


def test_import_pdf_table_no_continuation_warning_for_a_single_page_table(simple_pdf):
    result = import_pdf_table(simple_pdf, "single_page_pdf")
    assert result["continuation_pages"] == []
    assert result["stitched"] is False
    assert "stitched" not in result["note"]


def test_import_pdf_table_rejects_gold_prefixed_name(simple_pdf):
    with pytest.raises(ValueError, match="reserved"):
        import_pdf_table(simple_pdf, "gold_fake")


def test_import_pdf_table_missing_file():
    with pytest.raises(FileNotFoundError):
        import_pdf_table("no_such_file.pdf", "x")


def test_import_pdf_table_file_too_large(simple_pdf, monkeypatch):
    import src.agent.pdf_import as pi

    monkeypatch.setattr(pi, "MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="MB"):
        import_pdf_table(simple_pdf, "x")


def test_import_pdf_table_too_many_pages(tmp_path, monkeypatch):
    import src.agent.pdf_import as pi

    p = tmp_path / "one_page.pdf"
    _make_pdf(p, [["A", "B"], ["1", "2"]])
    monkeypatch.setattr(pi, "MAX_PDF_PAGES", 0)

    with pytest.raises(ValueError, match="limit is 0"):
        import_pdf_table(p, "x")


def test_pdf_import_is_findable_via_list_imported_datasets(simple_pdf):
    import_pdf_table(simple_pdf, "hastane_pdf")

    from src.agent.data_import import list_imported_datasets
    names = {d["dataset_name"] for d in list_imported_datasets()["datasets"]}
    assert "hastane_pdf" in names


def test_pdf_import_joinable_via_lakehouse_query(simple_pdf, tmp_path):
    """The whole point of moving imports into lakehouse.duckdb: a table
    from a PDF and a curated gold_* table live in the same file and can
    be JOINed in one query."""
    import duckdb
    import src.agent.data_import as di

    con = duckdb.connect(str(di.LAKEHOUSE_DB_PATH))
    con.execute("CREATE TABLE gold_fact_long AS SELECT 'Ankara' AS Bolge, 100.0 AS Deger")
    con.close()

    import_pdf_table(simple_pdf, "hastane_pdf")

    con = duckdb.connect(str(di.LAKEHOUSE_DB_PATH), read_only=True)
    joined = con.execute(
        "SELECT h.Il, g.Deger FROM hastane_pdf h JOIN gold_fact_long g ON h.Il = g.Bolge"
    ).fetchall()
    con.close()

    assert joined == [("Ankara", 100.0)]


# --- data cleaning wired into the PDF path too ------------------------------

def test_pdf_import_excludes_subtotal_rows(tmp_path):
    p = tmp_path / "sanayi.pdf"
    _make_pdf(p, [
        ["Donem", "Endeks"], ["2024-01", "102,3"], ["2024-02", "103,1"], ["Toplam", "205,4"],
    ])

    result = import_pdf_table(p, "sanayi_pdf")
    assert result["data_quality"]["subtotal_rows_excluded"] == 1
    assert result["row_count"] == 2


def test_pdf_import_expands_abbreviated_figures():
    from src.agent.pdf_import import _table_to_dataframe

    table = [["Sirket", "Ciro"], ["A", "1.5M"], ["B", "2.3B"]]
    df, report = _table_to_dataframe(table)
    assert "Ciro" in report["numeric_columns_recovered"]
    assert list(df["Ciro"]) == [1_500_000.0, 2_300_000_000.0]


def test_pdf_import_normalizes_null_tokens():
    from src.agent.pdf_import import _table_to_dataframe

    table = [["Il", "Deger"], ["Ankara", "42"], ["Istanbul", "-"], ["Izmir", "N/A"]]
    df, report = _table_to_dataframe(table)
    assert report["null_tokens_normalized"] == 2
    assert df["Deger"].isna().sum() == 2
