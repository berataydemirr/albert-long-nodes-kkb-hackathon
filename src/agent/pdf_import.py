"""Extract a table from a PDF and load it into the lakehouse.

An earlier pass in this project ruled PDF table extraction out entirely:
`pypdf` (the only PDF library that was a dependency at the time) reads
PDF TEXT, not table structure, and camelot/tabula need a Java/Ghostscript
runtime this project does not carry. `pdfplumber` changes that - it is
pure Python, already proven against a real BIST-bulletin-style PDF while
building this, and finds tables either from an actual grid of ruling
lines or, failing that, from aligned whitespace gaps between words. It
still cannot recover a table that has no visible structure at all (a
scanned image, a merged multi-line header, columns separated only by
inconsistent spacing) - see _table_to_dataframe below for exactly what
gets rejected rather than silently mis-imported.

Shares its write path with data_import.py (_write_dataset) rather than
a second copy of the collision-check/registry/retry logic: only the
PDF -> DataFrame step here is genuinely PDF-specific.
"""
from collections import Counter
from pathlib import Path

import pandas as pd
import pdfplumber

from src.agent.data_cleaning import is_null_token, is_subtotal_label, normalize_numeric_text, repair_mojibake
from src.agent.data_import import MAX_FILE_BYTES, _infer_column, _validate_dataset_name, _write_dataset
from src.common.logging_setup import get_logger

logger = get_logger("pdf_import")

MAX_PDF_PAGES = 30  # a runaway multi-hundred-page PDF should not be extracted whole


def _check_pdf_open(path: Path):
    if not path.exists():
        logger.error(f"_check_pdf_open: file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        logger.warning(f"_check_pdf_open: file too large ({size} bytes)")
        raise ValueError(f"File is {size / 1_048_576:.1f} MB; the limit is {MAX_FILE_BYTES / 1_048_576:.0f} MB.")
    pdf = pdfplumber.open(path)
    if len(pdf.pages) > MAX_PDF_PAGES:
        pdf.close()
        raise ValueError(f"PDF has {len(pdf.pages)} page(s); the limit is {MAX_PDF_PAGES}.")
    return pdf


def list_pdf_tables(path: str | Path) -> dict:
    """Preview: how many tables pdfplumber finds on each page and each
    one's shape, so a caller can pick page/table_index before importing -
    the PDF equivalent of list_sheets() for Excel."""
    path = Path(path)
    logger.info(f"list_pdf_tables: start path={path}")
    pdf = _check_pdf_open(path)
    try:
        found = []
        for page_no, page in enumerate(pdf.pages):
            for t_idx, table in enumerate(page.extract_tables()):
                n_rows = len(table)
                n_cols = max((len(r) for r in table), default=0)
                found.append({"page": page_no, "table_index": t_idx, "row_count": n_rows, "column_count": n_cols})
        logger.info(f"list_pdf_tables: {path.name} -> {len(found)} table(s) across {len(pdf.pages)} page(s)")
        return {"page_count": len(pdf.pages), "tables": found}
    finally:
        pdf.close()


def _table_to_dataframe(table: list[list]) -> tuple[pd.DataFrame, dict]:
    """pdfplumber returns a raw grid of strings/None - first row is
    treated as the header. Rejected outright rather than guessed at: a
    duplicate or blank header column (pdfplumber's most common failure
    mode, usually from a merged cell in the source table) means the
    extraction is not trustworthy, not that it needs a best-effort fix.

    Returns (dataframe, cleaning_report) - the same shared
    src/agent/data_cleaning.py normalizations data_import.py applies to
    CSV/Excel (Turkish/abbreviated numbers, null-token variants, subtotal
    rows, mojibake) run here too, so a PDF import is not the one path
    left unclean."""
    if not table or len(table) < 2:
        raise ValueError("Extracted table has no data rows (only a header, or nothing at all).")
    header = [repair_mojibake(("" if c is None else str(c).strip())) for c in table[0]]
    if any(h == "" for h in header):
        raise ValueError(
            "The extracted table has a blank column header - pdfplumber likely mis-split a merged "
            "cell. Try a different page/table_index, or a cleaner export of the same table."
        )
    if len(header) != len(set(header)):
        dupes = sorted(name for name, count in Counter(header).items() if count > 1)
        raise ValueError(f"Duplicate column name(s) in the extracted table header: {dupes}.")

    rows = [[repair_mojibake(("" if c is None else str(c).strip())) for c in row] for row in table[1:]]
    df = pd.DataFrame(rows, columns=header)
    report = {"numeric_columns_recovered": [], "null_tokens_normalized": 0, "subtotal_rows_excluded": 0}

    # There is no cell dtype in a PDF - every value comes back as text.
    # normalize_numeric_text handles Turkish vs international separator
    # ambiguity AND abbreviated magnitudes ('1.5M', '2,3 milyar') - a
    # column where >90% of values parse as a number this way is
    # converted; anything below that bar is left as text rather than
    # forcing a bad conversion that would turn a footnote marker into NaN
    # silently mixed in with real values.
    for col in df.columns:
        is_null_mask = df[col].apply(is_null_token)
        if is_null_mask.any():
            report["null_tokens_normalized"] += int(is_null_mask.sum())
            df.loc[is_null_mask, col] = None

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted = non_null.apply(normalize_numeric_text)
        if converted.notna().mean() > 0.9:
            report["numeric_columns_recovered"].append(str(col))
            df[col] = df[col].apply(lambda v: normalize_numeric_text(v) if v is not None else None)

    if len(df.columns) > 0:
        first_col = df.columns[0]
        subtotal_mask = df[first_col].apply(lambda v: isinstance(v, str) and is_subtotal_label(v))
        if subtotal_mask.any():
            report["subtotal_rows_excluded"] = int(subtotal_mask.sum())
            df = df[~subtotal_mask].reset_index(drop=True)

    return df, report


def import_pdf_table(path: str | Path, dataset_name: str, page: int = 0, table_index: int = 0,
                     source_label: str | None = None) -> dict:
    """Extract one table from one page of a PDF and load it into the
    lakehouse under dataset_name - same naming/collision/registry rules
    as import_dataset (data_import.py), just with a PDF-specific
    extraction step in front of the same write path."""
    path = Path(path)
    dataset_name = _validate_dataset_name(dataset_name)
    display_name = source_label or path.name
    logger.info(f"import_pdf_table: start path={path}, page={page}, table_index={table_index}, dataset_name={dataset_name!r}")

    pdf = _check_pdf_open(path)
    try:
        if page < 0 or page >= len(pdf.pages):
            raise ValueError(f"page={page} out of range; this PDF has {len(pdf.pages)} page(s) (0-indexed).")
        tables = pdf.pages[page].extract_tables()
        if not tables:
            raise ValueError(f"No table found on page {page}. Call list_pdf_tables first to see what pdfplumber can find.")
        if table_index < 0 or table_index >= len(tables):
            raise ValueError(f"table_index={table_index} out of range; page {page} has {len(tables)} table(s).")
        raw_table = tables[table_index]
        header = [("" if c is None else str(c).strip()) for c in raw_table[0]] if raw_table else []

        # A table that continues onto another page (the same header
        # repeated, e.g. via reportlab's own repeatRows) is NOT stitched
        # together - only the requested page/table_index is imported.
        # Found live: a 60-row test table spanning two pages imported only
        # the 37 rows that were actually on page 0, silently. Detected
        # here by an identical header appearing on a different page, and
        # surfaced as a note rather than fixed automatically - guessing
        # which other tables belong to the same logical table is a real
        # risk of its own (page 2 could just as easily be an unrelated
        # table that happens to share column names).
        continuation_pages = []
        if header:
            for other_page_no, other_page in enumerate(pdf.pages):
                if other_page_no == page:
                    continue
                for other_table in other_page.extract_tables():
                    other_header = [("" if c is None else str(c).strip()) for c in other_table[0]] if other_table else []
                    if other_header == header:
                        continuation_pages.append(other_page_no)
                        break
    finally:
        pdf.close()

    df, cleaning_report = _table_to_dataframe(raw_table)
    columns = [_infer_column(df[col]) for col in df.columns]

    imported_at = _write_dataset(dataset_name, display_name, df)
    note = (
        "Extracted from a PDF via layout-based table detection (pdfplumber), not a real "
        "spreadsheet grid - spot-check a few values against the source page before relying on "
        "this; a low-quality scan or an unusual layout can misread a cell without raising an error."
    )
    if continuation_pages:
        note += (
            f" WARNING: page(s) {continuation_pages} have a table with an IDENTICAL header - this "
            f"table likely continues there, and only page {page} was imported. Import each page "
            f"separately (same dataset_name replaces, so import into a different name and UNION, "
            f"or ask for the full picture across all of them) if the complete table is needed."
        )
    result = {
        "dataset_name": dataset_name,
        "imported_at": imported_at,
        "file_name": display_name,
        "page": page,
        "table_index": table_index,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "data_quality": cleaning_report,
        "continuation_pages": continuation_pages,
        "note": note,
    }
    logger.info(f"import_pdf_table: done, {dataset_name!r} <- {len(df)} rows x {len(df.columns)} cols, continuation_pages={continuation_pages}")
    return result
