"""Tests for PDF table extraction/import (src/agent/pdf_import.py).

Real PDFs are built here with reportlab (test-only dependency) rather
than hand-crafted bytes or a mocked pdfplumber - pdfplumber's table
detection depends on actual vector line graphics on the page, so a fake
that skips PDF rendering entirely would not catch a real regression in
how pdfplumber sees the file.
"""
import pandas as pd
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


def _make_pdf(path, rows, page_break_before=None):
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    elements = []
    if page_break_before:
        from reportlab.platypus import PageBreak, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        elements.append(Paragraph("Not a table - just text.", getSampleStyleSheet()["Normal"]))
        elements.append(PageBreak())
    t = Table(rows)
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
