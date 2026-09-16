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
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pdfplumber

from config.settings import PROJECT_ROOT
from src.agent.data_cleaning import is_null_token, is_subtotal_label, normalize_numeric_text, repair_mojibake
from src.agent.data_import import MAX_FILE_BYTES, _infer_column, _validate_dataset_name, _write_dataset
from src.common.logging_setup import get_logger

logger = get_logger("pdf_import")

MAX_PDF_PAGES = 60  # a runaway multi-hundred-page PDF should not be extracted whole
# Raised from 30: a real Hazine ve Maliye Bakanligi "Kamu Borc Yonetimi
# Raporu" (downloaded live, Aralik 2025 edition) is 47 pages and was
# rejected outright at 30 - a routine, real government report, not an
# adversarial input. 60 keeps a real ceiling against a runaway file while
# covering this and similar multi-page annual/quarterly reports.

# OCR fallback for a page pdfplumber finds NO table structure on at all -
# a scanned page or, the real case this was built against, a chart/graph
# image export (verified live: a real TCMB "seyahat gelirleri" chart PDF
# has 0 chars/lines/rects and 1 image on its only page). The chart's own
# DATA (the line's actual values) is a picture, not text, and OCR cannot
# recover it - but the surrounding text very often can be read: title,
# the real Seri_Kod values and their descriptions, source/metadata links.
# Confirmed live against that exact file: OCR correctly recovered
# 'TP.SGEGI.K1-8', its description, and 'Veri Kaynagi: TURKIYE ISTATISTIK
# KURUMU' - genuinely useful context an agent can act on, even though no
# row/column table can be built from a chart. This NEVER fabricates a
# table: import_pdf_table still raises when there is no real table, OCR
# only enriches that error message with whatever text it could read.
#
# Language data is NOT bundled in the repo (tessdata/ is gitignored, ~8MB
# of binary model files) - a teammate without it set up gets the exact
# same plain rejection as before OCR existed, never a crash. To set it up:
# install Tesseract (winget install UB-Mannheim.TesseractOCR on Windows,
# or `apt install tesseract-ocr` / `brew install tesseract`), then copy
# eng.traineddata from its install and download tur.traineddata from
# https://github.com/tesseract-ocr/tessdata_fast into this project's
# tessdata/ folder.
_TESSDATA_DIR = PROJECT_ROOT / "tessdata"
_TESSDATA_DIR_REAL = _TESSDATA_DIR  # stable reference tests restore after monkeypatching _TESSDATA_DIR off
_WINDOWS_TESSERACT_FALLBACK = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_OCR_TEXT_MAX_CHARS = 1500


def _ocr_page_text(page) -> str | None:
    """Best-effort only - returns None (never raises) whenever Tesseract or
    its language data is not available, so OCR being unset up never breaks
    the plain-rejection behavior it is layered on top of."""
    if not _TESSDATA_DIR.exists():
        return None
    try:
        import pytesseract
    except ImportError:
        logger.info("_ocr_page_text: pytesseract not installed, skipping OCR fallback")
        return None

    config = f"--tessdata-dir {_TESSDATA_DIR}"

    def _run() -> str:
        image = page.to_image(resolution=200).original
        return pytesseract.image_to_string(image, lang="tur+eng", config=config)

    try:
        text = _run()
    except pytesseract.TesseractNotFoundError:
        # 'tesseract' not on PATH yet (a fresh install can need a shell/
        # session restart before PATH updates take effect) - try the
        # standard Windows install location once before giving up.
        if shutil.which("tesseract") or not Path(_WINDOWS_TESSERACT_FALLBACK).exists():
            logger.info("_ocr_page_text: tesseract binary not found, skipping OCR fallback")
            return None
        try:
            pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT_FALLBACK
            text = _run()
        except Exception as e:
            logger.warning(f"_ocr_page_text: OCR failed via Windows fallback path: {e}")
            return None
    except Exception as e:
        logger.warning(f"_ocr_page_text: OCR failed unexpectedly: {e}")
        return None

    text = text.strip()
    if not text:
        return None
    logger.info(f"_ocr_page_text: recovered {len(text)} char(s) of OCR text")
    return text[:_OCR_TEXT_MAX_CHARS]


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


def _flatten_pdf_merged_header(header: list[str], sub_header: list[str]) -> list[str] | None:
    """4. Birleştirilmiş Excel başlıkları - the PDF equivalent: a year
    (or any top-level label) spanning several sub-columns (quarters,
    months) leaves pdfplumber's raw grid with the year on the merge's
    first cell and '' on the rest, exactly like Excel's 'Unnamed: N'
    pattern (see data_import.py's _flatten_merged_excel_header, the same
    idea applied here since a PDF table has no pandas column names to key
    off). Real case found live: a Hazine ve Maliye Bakanligi budget table
    with 'Milyon TL' / blank / '2025' / blank*11 as row 0 and the real
    month abbreviations as row 1. Returns None (caller keeps the original
    blank-header rejection) if forward-filling still leaves a blank or
    duplicate column - never guesses past that point."""
    new_header = []
    last_real_top = None
    for i, cell in enumerate(header):
        top = cell if cell else last_real_top
        if cell:
            last_real_top = cell
        sub = sub_header[i].strip() if i < len(sub_header) and sub_header[i] else ""
        if top and sub:
            new_header.append(f"{top}_{sub}")
        elif sub:
            new_header.append(sub)
        else:
            new_header.append(top or "")

    if any(h == "" for h in new_header) or len(new_header) != len(set(new_header)):
        return None
    return new_header


def _table_to_dataframe(table: list[list]) -> tuple[pd.DataFrame, dict]:
    """pdfplumber returns a raw grid of strings/None - first row is
    treated as the header. A blank header cell is first tried as a
    merged top-level label spanning sub-columns, same as Excel's merged
    header (see _flatten_pdf_merged_header); if that does not fully
    resolve it, rejected outright rather than guessed at further - a
    duplicate or still-blank header column means the extraction is not
    trustworthy, not that it needs a best-effort fix.

    Returns (dataframe, cleaning_report) - the same shared
    src/agent/data_cleaning.py normalizations data_import.py applies to
    CSV/Excel (Turkish/abbreviated numbers, null-token variants, subtotal
    rows, mojibake) run here too, so a PDF import is not the one path
    left unclean."""
    if not table or len(table) < 2:
        raise ValueError("Extracted table has no data rows (only a header, or nothing at all).")
    header = [repair_mojibake(("" if c is None else str(c).strip())) for c in table[0]]
    data_rows = table[1:]
    merged_header_used = False
    if any(h == "" for h in header):
        sub_header_raw = table[1] if len(table) > 2 else []
        sub_header = [repair_mojibake("" if c is None else str(c).strip()) for c in sub_header_raw]
        flattened = _flatten_pdf_merged_header(header, sub_header)
        if flattened is not None and len(table) > 2:
            header = flattened
            data_rows = table[2:]
            merged_header_used = True
        else:
            raise ValueError(
                "The extracted table has a blank column header - pdfplumber likely mis-split a merged "
                "cell. Try a different page/table_index, or a cleaner export of the same table."
            )
    if len(header) != len(set(header)):
        dupes = sorted(name for name, count in Counter(header).items() if count > 1)
        raise ValueError(f"Duplicate column name(s) in the extracted table header: {dupes}.")

    rows = [[repair_mojibake(("" if c is None else str(c).strip())) for c in row] for row in data_rows]
    df = pd.DataFrame(rows, columns=header)
    report = {
        "numeric_columns_recovered": [], "null_tokens_normalized": 0, "subtotal_rows_excluded": 0,
        "merged_header_flattened": merged_header_used,
    }

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
                     source_label: str | None = None, stitch_continuation_pages: bool = True) -> dict:
    """Extract one table from a PDF and load it into the lakehouse under
    dataset_name - same naming/collision/registry rules as import_dataset
    (data_import.py), just with a PDF-specific extraction step in front
    of the same write path.

    stitch_continuation_pages (default True): a table split across pages
    (the same header repeated, e.g. via reportlab's own repeatRows, or a
    real multi-page BDDK/kâr-raporu-style bulletin) is detected by an
    IDENTICAL header appearing on another page, and its data rows are
    concatenated onto the requested page's - found live, a 60-row test
    table spanning two pages silently imported only the 37 rows that
    were on the requested page. Only a page whose header matches EXACTLY
    is stitched in, in page order; a page with a differently-shaped table
    that happens to sit nearby is never guessed into the same table. Pass
    False to import only the literal requested page (the old behavior),
    e.g. when two genuinely different tables happen to share a header."""
    path = Path(path)
    dataset_name = _validate_dataset_name(dataset_name)
    display_name = source_label or path.name
    logger.info(
        f"import_pdf_table: start path={path}, page={page}, table_index={table_index}, "
        f"dataset_name={dataset_name!r}, stitch_continuation_pages={stitch_continuation_pages}"
    )

    pdf = _check_pdf_open(path)
    try:
        if page < 0 or page >= len(pdf.pages):
            raise ValueError(f"page={page} out of range; this PDF has {len(pdf.pages)} page(s) (0-indexed).")
        tables = pdf.pages[page].extract_tables()
        if not tables:
            ocr_text = _ocr_page_text(pdf.pages[page])
            if ocr_text:
                raise ValueError(
                    f"No table found on page {page} - pdfplumber sees no ruling-line or "
                    f"whitespace-aligned grid, likely a chart/scanned image rather than a real "
                    f"table (its own data points, if any, cannot be recovered this way). OCR did "
                    f"read text on the page though, which may identify what it actually shows:\n\n"
                    f"{ocr_text}"
                )
            raise ValueError(f"No table found on page {page}. Call list_pdf_tables first to see what pdfplumber can find.")
        if table_index < 0 or table_index >= len(tables):
            raise ValueError(f"table_index={table_index} out of range; page {page} has {len(tables)} table(s).")
        raw_table = tables[table_index]
        header = [("" if c is None else str(c).strip()) for c in raw_table[0]] if raw_table else []

        # (page_no, data_rows) for every page with a table whose header
        # matches EXACTLY, requested page included - collected once, in
        # ascending page order, and reused for both the continuation-page
        # report and the actual stitch below (no re-extraction).
        matching_pages: list[tuple[int, list]] = [(page, raw_table[1:])]
        if header:
            for other_page_no, other_page in enumerate(pdf.pages):
                if other_page_no == page:
                    continue
                for other_table in other_page.extract_tables():
                    other_header = [("" if c is None else str(c).strip()) for c in other_table[0]] if other_table else []
                    if other_header == header:
                        matching_pages.append((other_page_no, other_table[1:]))
                        break
        continuation_pages = sorted(p for p, _ in matching_pages if p != page)

        if stitch_continuation_pages and continuation_pages:
            matching_pages.sort(key=lambda pr: pr[0])
            raw_table = [raw_table[0]] + [row for _, rows in matching_pages for row in rows]
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
    if continuation_pages and stitch_continuation_pages:
        note += (
            f" Table continued on page(s) {continuation_pages} (identical header) - stitched into "
            f"this import automatically; row_count below covers all of them."
        )
    elif continuation_pages:
        note += (
            f" WARNING: page(s) {continuation_pages} have a table with an IDENTICAL header and were "
            f"NOT stitched in (stitch_continuation_pages=False) - only page {page} was imported."
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
        "stitched": bool(continuation_pages and stitch_continuation_pages),
        "note": note,
    }
    logger.info(f"import_pdf_table: done, {dataset_name!r} <- {len(df)} rows x {len(df.columns)} cols, continuation_pages={continuation_pages}")
    return result
