"""Import a user-supplied CSV/Excel file and let the agent query it.

WHY THIS EXISTS:
Every other tool in this project answers questions about data the team
collected and vetted ahead of time (BDDK, EVDS, TUIK). Team feedback asked
for the opposite capability: a user hands the system an arbitrary file at
runtime - possibly from a domain nobody built a collector for - and the
agent should be able to read, analyse, and JOIN it against the existing
lakehouse in the same conversation.

WHAT THIS DELIBERATELY IS NOT:
A PDF table extractor. See pdf_import.py for that (pdfplumber-based,
still CSV/Excel-adjacent but its own module since PDF table structure has
to be reconstructed from text positions rather than read from a real grid).

DESIGN (revised from an earlier, more isolated version):
Every import and live refresh now lands directly in lakehouse.duckdb, in
its own table named after the dataset, so a single lakehouse_query call
can JOIN it against gold_fact_long and the other curated tables - this is
the actual point of "genelleştirilebilir olmalı, ilişkilendirilebilir
olmalı" from the team feedback: an isolated second database file cannot
be joined against anything.

The earlier design put uploads in a completely separate DuckDB file
specifically to protect the curated gold layer from unverified data. That
protection is kept, just moved from "a different file" to two narrower
guarantees that hold within the one shared file:
  1. dataset_name can never collide with a real gold_* table or the
     registry table itself (checked both statically, in
     _validate_dataset_name, and live against the connection's own
     SHOW TABLES before every write).
  2. every imported table is tracked in _import_registry, and anything
     built from it - lakehouse_schema's own catalog output, an answer
     that cites it - is expected to say plainly that it is user-supplied
     and unverified, unlike a gold_* series.
Two connections briefly wanting the same file (an import landing while a
lakehouse_query is mid-flight) is now possible where it wasn't before;
see db_utils.connect_with_retry for how that race is handled.

Column dtypes are inferred defensively: pandas' own type inference is
good at splitting numbers from text but does not distinguish "actually a
date" from "a string that happens to look like one" reliably across
locales, so it is checked explicitly here rather than trusted blindly -
important for a domain-agnostic tool, since a user's file has no unit or
type metadata the way the vetted BDDK/EVDS tables do.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from config.settings import LAKEHOUSE_DB_PATH
from config.tuik_series import SERIES as TUIK_REGISTERED_SERIES
from src.agent.db_utils import connect_with_retry
from src.agent.sql_guard import run_guarded_select
from src.common.logging_setup import get_logger

logger = get_logger("data_import")

REGISTRY_TABLE = "_import_registry"
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RESERVED_PREFIX = "gold_"
MAX_ROWS_PER_IMPORT = 200_000
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB - generous for a demo CSV/XLSX, small enough to import in-process
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Per-column type inference. pandas already separates numeric from
# object dtype; this narrows "object" further into text vs. date, and
# reports what was actually decided so a wrong guess is visible rather
# than silent.
_DATE_LIKE_RE = re.compile(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")


def _db(read_only: bool) -> duckdb.DuckDBPyConnection:
    if read_only and not LAKEHOUSE_DB_PATH.exists():
        # A read against a lakehouse that has never been built (or, in a
        # test, a fresh tmp_path with nothing imported yet) is a normal
        # "nothing here yet" state, not an error - list_imported_datasets
        # relies on this to return an empty list rather than raising.
        logger.info(f"_db: {LAKEHOUSE_DB_PATH} does not exist yet")
        raise FileNotFoundError(f"Lakehouse database not found: {LAKEHOUSE_DB_PATH}.")
    if not read_only:
        LAKEHOUSE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"_db: opening {LAKEHOUSE_DB_PATH} (read_only={read_only})")
    return connect_with_retry(LAKEHOUSE_DB_PATH, read_only=read_only)


def _validate_dataset_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        logger.warning(f"_validate_dataset_name: rejected {name!r}")
        raise ValueError(
            f"dataset_name must match {_NAME_RE.pattern} (letters, digits, underscore, "
            f"starting with a letter or underscore, max 64 chars). Got: {name!r}"
        )
    if name == REGISTRY_TABLE:
        raise ValueError(f"dataset_name {name!r} is reserved.")
    if name.startswith(_RESERVED_PREFIX):
        raise ValueError(
            f"dataset_name {name!r} is reserved for the curated gold layer "
            f"(names starting with {_RESERVED_PREFIX!r} are not allowed for an import)."
        )
    return name


def _reject_curated_collision(con: duckdb.DuckDBPyConnection, dataset_name: str) -> None:
    """A dataset_name that already names a REAL table - one not tracked in
    _import_registry, i.e. a curated gold_* table or some other object -
    must never be silently overwritten by CREATE OR REPLACE TABLE. The
    static gold_ prefix check in _validate_dataset_name catches the
    common case; this catches everything else (a table with no reserved
    prefix, a future table this pass doesn't know about) by checking the
    connection's own SHOW TABLES rather than a hardcoded list."""
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if dataset_name not in tables:
        return
    previously_imported: set[str] = set()
    if REGISTRY_TABLE in tables:
        previously_imported = {row[0] for row in con.execute(f"SELECT dataset_name FROM {REGISTRY_TABLE}").fetchall()}
    if dataset_name not in previously_imported:
        logger.warning(f"_reject_curated_collision: {dataset_name!r} already exists and is not a tracked import")
        raise ValueError(
            f"dataset_name {dataset_name!r} already names a table in the lakehouse that was not "
            f"created by an import - refusing to overwrite it. Choose a different name."
        )


def _read_table(path: Path, sheet: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        logger.debug(f"_read_table: reading CSV {path}")
        try:
            # encoding='utf-8-sig' quietly handles a leading BOM (common
            # from Excel's own "CSV UTF-8" export) that would otherwise
            # land inside the first column's name and make it
            # un-referenceable by SQL.
            return pd.read_csv(path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError as exc:
            logger.warning(f"_read_table: {path} has no parseable content")
            raise ValueError(f"'{path.name}' is empty or has no columns.") from exc
    if suffix in (".xlsx", ".xls"):
        logger.debug(f"_read_table: reading Excel {path}, sheet={sheet!r}")
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        # Hidden-row/column exclusion runs BEFORE header flattening, on
        # the raw pandas read, where a data row/column index reliably
        # lines up with the underlying Excel row/column number - doing it
        # after flattening (which can drop the first data row entirely,
        # see _flatten_merged_excel_header) would shift that alignment by
        # one for no benefit.
        hidden_rows, hidden_cols = _hidden_excel_rows_and_columns(path, sheet)
        if hidden_cols:
            keep = [c for i, c in enumerate(df.columns) if i not in hidden_cols]
            dropped = len(df.columns) - len(keep)
            df = df[keep]
            logger.info(f"_read_table: excluded {dropped} hidden column(s)")
        if hidden_rows:
            before = len(df)
            df = df[~df.index.isin(hidden_rows)].reset_index(drop=True)
            logger.info(f"_read_table: excluded {before - len(df)} hidden row(s)")
        return _flatten_merged_excel_header(df)
    logger.warning(f"_read_table: unsupported extension {suffix!r}")
    raise ValueError(f"Unsupported file type {suffix!r}. Supported: {SUPPORTED_EXTENSIONS}.")


def _hidden_excel_rows_and_columns(path: Path, sheet: str | int | None) -> tuple[set[int], set[int]]:
    """7. Gizli satır ve sütun içeren dosyalar: openpyxl's non-streaming
    worksheet (read_only=False - the streaming reader used elsewhere does
    not expose row_dimensions/column_dimensions) reports which physical
    rows/columns are hidden. A real bilanço export often hides a row left
    over from a broken formula rather than deleting it - pandas' own
    reader has no concept of 'hidden' at all and would otherwise import
    it as an ordinary visible observation.

    Returns 0-indexed positions matching how pandas.read_excel(header=0)
    sees the data: row 0 = the first row AFTER the header (Excel row 2),
    column 0 = the first column (Excel column A).

    Legacy .xls (pre-2007 binary format): openpyxl cannot open this format
    at all (it raises InvalidFileException - confirmed live against a real
    TBB .xls export) - only .xlsx has an OOXML zip structure for openpyxl
    to read. xlrd 2.x, the library that CAN still open .xls, does not
    expose per-row/per-column hidden flags in a comparable way. Rather
    than fail the whole import over a feature .xls cannot support here,
    hidden-row/column exclusion is simply skipped for .xls (reported as
    0/0, same shape as when a file has no hidden rows at all)."""
    if path.suffix.lower() == ".xls":
        return set(), set()

    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
    try:
        ws = wb[sheet] if isinstance(sheet, str) else wb.worksheets[sheet or 0]
        hidden_rows = {
            excel_row_no - 2  # -2: 1-indexed Excel row, minus the header row
            for excel_row_no, dim in ws.row_dimensions.items()
            if dim.hidden and excel_row_no >= 2
        }
        hidden_cols = {
            openpyxl.utils.column_index_from_string(col_letter) - 1
            for col_letter, dim in ws.column_dimensions.items()
            if dim.hidden
        }
        return hidden_rows, hidden_cols
    finally:
        wb.close()


_UNNAMED_COL_RE = re.compile(r"^Unnamed: \d+$")


def _flatten_merged_excel_header(df: pd.DataFrame) -> pd.DataFrame:
    """4. Birleştirilmiş Excel başlıkları: a merged header cell (e.g. a
    year spanning several month sub-columns, kredi kartı harcamaları-
    style) reads back from pandas with the year on the first column and
    'Unnamed: N' on the rest - the real sub-header (the month names)
    lands in the first DATA row instead, since Excel only records where
    a merge starts and pandas has no way to know two header levels were
    meant. Detected purely by pandas' own 'Unnamed: N' naming convention
    for a blank header cell (whether from a merge or a genuinely empty
    one) - if none are present, the DataFrame is returned unchanged, so
    this never touches an ordinary single-row-header file."""
    if not any(_UNNAMED_COL_RE.match(str(c)) for c in df.columns):
        return df
    if len(df) == 0:
        return df

    sub_header_row = df.iloc[0]
    new_columns = []
    tops_used: list[str] = []
    last_real_top = None
    for col in df.columns:
        is_unnamed = bool(_UNNAMED_COL_RE.match(str(col)))
        top = last_real_top if is_unnamed else str(col)
        if not is_unnamed:
            last_real_top = top
        tops_used.append(top or "")
        sub_value = sub_header_row[col]
        sub_str = "" if pd.isna(sub_value) else str(sub_value).strip()
        if top and sub_str:
            new_columns.append(f"{top}_{sub_str}")
        elif sub_str:
            new_columns.append(sub_str)
        else:
            new_columns.append(top or str(col))

    # A genuine "year over months" merge always leaves a distinct sub_str
    # per column, so new_columns comes out unique in that case. Real bug
    # found live against a TBB .xls export: when the "merge" is actually
    # a single full-width title row over two UNRELATED side-by-side
    # tables (not this function's target shape at all - see issue #19),
    # every column shares the SAME top (one merge anchored at column 0
    # spans the entire header) with no sub_str to distinguish them, and
    # new_columns collides - `df[col]` for a duplicated name returns a
    # DataFrame slice instead of a Series and crashes every downstream
    # caller with a cryptic TypeError.
    #
    # A real "year over months" merge never hits this: it always keeps at
    # least one OTHER real top-level label distinct from the merged one
    # (e.g. a plain "Il" column before the year starts), so len(set(tops))
    # is always >= 2 there. A single top value used for literally every
    # column, combined with an actual name collision, is a much stronger
    # and more specific signal than the collision alone - reject outright
    # with an actionable message rather than silently deduplicate into
    # confusing "Title", "Title.1", "Title.2" columns that still don't
    # reflect the file's real two-tables-on-one-sheet layout.
    has_collision = len(new_columns) != len(set(new_columns))
    if has_collision and len(set(tops_used)) == 1:
        raise ValueError(
            "This sheet looks like it has more than one table side by side under a single "
            "full-width header cell, not one table with a merged 'year over sub-columns' "
            "header - every column resolved to the same header text "
            f"({tops_used[0]!r}), which cannot be a real column list. Re-export each table "
            "as its own sheet, or select just one table's cell range before uploading."
        )

    seen: dict[str, int] = {}
    deduped = []
    for name in new_columns:
        if name not in seen:
            seen[name] = 0
            deduped.append(name)
        else:
            seen[name] += 1
            deduped.append(f"{name}.{seen[name]}")

    flattened = df.iloc[1:].reset_index(drop=True).copy()
    flattened.columns = deduped
    logger.info(f"_flatten_merged_excel_header: merged header detected, recovered columns {deduped}")
    return flattened


def _raw_header(path: Path, sheet: str | int | None) -> list[str]:
    """The file's actual header row, before pandas silently deduplicates a
    repeated name (e.g. two 'Deger' columns become 'Deger' and 'Deger.1').
    That renaming hides a real problem - a user's file naming two columns
    the same thing is ambiguous, and importing it as if pandas' invented
    suffix were the real name would be reporting something the file
    itself does not say. Checked BEFORE the deduplicated read, using a
    separate raw pass, since by the time pd.read_csv/read_excel returns,
    the evidence of the collision is already gone from df.columns."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            row = next(csv.reader(f), [])
        return row
    if suffix == ".xls":
        # Legacy binary format: openpyxl cannot open it at all (see
        # _hidden_excel_rows_and_columns for the same finding), so there
        # is no .merged_cells to inspect here either. Read the first row
        # through pandas/xlrd instead - a merged header cell still shows
        # up as a blank ('' after the None->"" cast below) rather than a
        # forward-filled duplicate, same effect as the synthetic
        # placeholder below, just without a real merge range to anchor it
        # to (pandas' own 'Unnamed: N' columns handle the rest downstream,
        # same as for .xlsx - see _flatten_merged_excel_header). Each blank
        # cell still gets a unique synthetic placeholder, same reasoning as
        # the .xlsx branch below: two blank header cells are NOT the same
        # ambiguous name, and must not trip the duplicate-column rejection.
        row = list(pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, header=None, nrows=1).iloc[0])
        return [f"__blank_header_{i}__" if pd.isna(v) else str(v) for i, v in enumerate(row)]
    if suffix == ".xlsx":
        import openpyxl
        # read_only=False (not the streaming reader used elsewhere): only
        # the non-streaming worksheet exposes .merged_cells, needed next.
        wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
        try:
            ws = wb[sheet] if isinstance(sheet, str) else wb.worksheets[sheet or 0]
            row = list(next(ws.iter_rows(max_row=1, values_only=True), ()))
            # A merged header cell (a year spanning several month
            # sub-columns - see _flatten_merged_excel_header) leaves every
            # cell but the merge's own anchor as None. Left as-is, those
            # would read as several separate blank header cells and wrongly
            # trip the duplicate-column rejection below in detect_schema -
            # forward-filling the anchor's value instead would ALSO trip it
            # (now several identical values instead of several blanks), so
            # neither is right for a check whose job is "did the user
            # accidentally name two columns the same thing". Each
            # continuation cell gets a unique synthetic placeholder instead
            # - never a duplicate of anything - and _flatten_merged_excel_header
            # (given the real merged-cell ranges directly, not this raw
            # header) does the actual header reconstruction.
            for merged_range in ws.merged_cells.ranges:
                if merged_range.min_row <= 1 <= merged_range.max_row:
                    for col_idx in range(merged_range.min_col + 1, merged_range.max_col + 1):
                        if col_idx - 1 < len(row):
                            row[col_idx - 1] = f"__merge_continuation_{col_idx}__"
            return ["" if v is None else str(v) for v in row]
        finally:
            wb.close()
    return []


def list_sheets(path: str | Path) -> list[str]:
    """Excel-only: the sheet names in the workbook, for a caller that
    needs to ask the user which one to import (see the feedback's own
    "sheet alignment" concern - a workbook is not always one table)."""
    path = Path(path)
    if path.suffix.lower() not in (".xlsx", ".xls"):
        return []
    sheets = pd.ExcelFile(path).sheet_names
    logger.info(f"list_sheets: {path.name} -> {sheets}")
    return sheets


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Runs every src/agent/data_cleaning.py normalization over a
    freshly-read DataFrame, BEFORE schema detection or writing - so
    detect_schema and import_dataset always agree on what the file
    actually contains. Every change is reported back, never silent: a
    dropped subtotal row or a column recovered from Turkish/abbreviated
    number text is exactly the kind of thing a caller needs to see, the
    same discipline as the pre-existing row-truncation report."""
    from src.agent.data_cleaning import is_null_token, is_subtotal_label, normalize_numeric_text, parse_flexible_date, repair_mojibake

    df = df.copy()
    report = {
        "null_tokens_normalized": 0,
        "numeric_columns_recovered": [],
        "date_columns_normalized": [],
        "subtotal_rows_excluded": 0,
        "mojibake_repaired_columns": [],
    }

    repaired_columns = [repair_mojibake(str(c)) for c in df.columns]
    if repaired_columns != [str(c) for c in df.columns]:
        report["mojibake_repaired_columns"].append("(column headers)")
    df.columns = repaired_columns

    for col in df.columns:
        # NOT "dtype == object": pandas 3.x defaults a plain text column
        # to its own dedicated 'str' dtype, not legacy 'object' - found
        # live, this check silently skipped every text column in this
        # environment, so none of the cleaning below ever ran. Skip only
        # columns that are ALREADY numeric/bool/datetime; anything else
        # (object OR the new string dtype) is a candidate for cleaning.
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Compare only the STRING values that actually went through
        # repair_mojibake, not the whole series via .equals(): a
        # non-string value round-tripped through .apply() can come back
        # as a narrower dtype (float64 instead of object) even though no
        # value actually changed, and .equals() is sensitive to that
        # dtype difference - found live, a column of plain floats got
        # reported as "mojibake repaired" for no real reason.
        original_strings = [v for v in df[col] if isinstance(v, str)]
        repaired_strings = [repair_mojibake(v) for v in original_strings]
        if repaired_strings != original_strings:
            report["mojibake_repaired_columns"].append(str(col))
            df[col] = df[col].apply(lambda v: repair_mojibake(v) if isinstance(v, str) else v)

        # "-", "N/A", ".." etc -> real NaN, before anything downstream
        # (numeric conversion, analysis) has to special-case them.
        is_null_mask = df[col].apply(lambda v: isinstance(v, str) and is_null_token(v))
        if is_null_mask.any():
            report["null_tokens_normalized"] += int(is_null_mask.sum())
            df.loc[is_null_mask, col] = None

        # Recover a numeric column pandas' own inference left as text
        # because it does not understand Turkish separators ('1.234,56')
        # or abbreviated magnitudes ('1.5M', '2,3 milyar') - only applied
        # when >90% of the non-null values actually parse, so a genuinely
        # mixed/text column is never silently mangled.
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted = non_null.apply(normalize_numeric_text)
        if converted.notna().mean() > 0.9:
            report["numeric_columns_recovered"].append(str(col))
            df[col] = df[col].apply(lambda v: normalize_numeric_text(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None)
            continue

        # 2. Karmaşık tarih formatları: a column mixing 'YYYY-MM-DD',
        # '2023 1. Çeyrek', or '01-Mart-2023' across sources (or even
        # within one uploaded file) is normalized to this project's
        # 'YYYY-MM' Period convention in place, so it can actually be
        # JOINed against a gold_* series on Period without the caller
        # having to reconcile formats by hand. Only applied when >90% of
        # values parse, same discipline as the numeric recovery above.
        date_converted = non_null.apply(parse_flexible_date)
        if date_converted.notna().mean() > 0.9:
            report["date_columns_normalized"].append(str(col))
            df[col] = df[col].apply(lambda v: parse_flexible_date(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None)

    # Subtotal-row exclusion, checked against the FIRST column only - the
    # conventional label/category position in a BDDK/TUIK-style table. A
    # subtotal marker anywhere else is data, not noise, and left alone.
    if len(df.columns) > 0:
        first_col = df.columns[0]
        subtotal_mask = df[first_col].apply(lambda v: isinstance(v, str) and is_subtotal_label(v))
        if subtotal_mask.any():
            report["subtotal_rows_excluded"] = int(subtotal_mask.sum())
            df = df[~subtotal_mask].reset_index(drop=True)

    return df, report


def _infer_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    null_count = int(series.isna().sum())

    if pd.api.types.is_bool_dtype(series):
        kind = "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        kind = "date"
    else:
        # object dtype: numeric-looking or date-looking strings are common
        # in a CSV export (no native typing) - checked explicitly rather
        # than left as "text", since a rate or amount column typed as text
        # is unusable by any analysis tool downstream.
        sample = non_null.astype(str).head(50)
        if len(sample) and sample.str.match(_DATE_LIKE_RE).mean() > 0.9:
            kind = "date_like_text"
        elif len(sample) and pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9:
            kind = "numeric_like_text"
        else:
            kind = "text"

    info = {
        "column": str(series.name),
        "inferred_type": kind,
        "null_count": null_count,
        "non_null_count": int(len(series) - null_count),
    }
    if kind in ("numeric",):
        info["min"] = None if non_null.empty else float(non_null.min())
        info["max"] = None if non_null.empty else float(non_null.max())
    else:
        distinct_sample = non_null.astype(str).unique()[:5]
        info["sample_values"] = list(distinct_sample)
    return info


def detect_schema(path: str | Path, sheet: str | int | None = None) -> dict:
    """Read a CSV/Excel file and report its columns, inferred types and a
    few sample values - without importing anything yet. Used both as the
    standalone `detect_schema` tool (a user can ask "what's in this file"
    before committing to an import) and internally by import_dataset."""
    path = Path(path)
    logger.info(f"detect_schema: start path={path}, sheet={sheet!r}")
    if not path.exists():
        logger.error(f"detect_schema: file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        logger.warning(f"detect_schema: file too large ({size} bytes)")
        raise ValueError(f"File is {size / 1_048_576:.1f} MB; the limit is {MAX_FILE_BYTES / 1_048_576:.0f} MB.")

    raw_header = _raw_header(path, sheet)
    if len(raw_header) != len(set(raw_header)):
        # Checked against the RAW header, not df.columns: pandas silently
        # renames a repeat ("Deger" / "Deger.1") during the parse below,
        # which would otherwise hide the collision entirely - found live,
        # a file with two "Deger" columns imported with no error and no
        # indication that the second column was not what its name claimed.
        from collections import Counter
        dupes = sorted(name for name, count in Counter(raw_header).items() if count > 1)
        logger.warning(f"detect_schema: duplicate column name(s) in the raw header: {dupes}")
        raise ValueError(f"Duplicate column name(s) in the file: {dupes}. Rename them and re-upload.")

    df = _read_table(path, sheet)
    df, cleaning_report = _clean_dataframe(df)
    if path.suffix.lower() in (".xlsx", ".xls"):
        # _read_table already excluded these before this point - a second,
        # cheap (dimensions only, not the data grid) call just for the
        # count, so it can be reported the same way as every other
        # cleaning change rather than only ever appearing in the log.
        hidden_rows, hidden_cols = _hidden_excel_rows_and_columns(path, sheet)
        cleaning_report["hidden_rows_excluded"] = len(hidden_rows)
        cleaning_report["hidden_columns_excluded"] = len(hidden_cols)
    logger.info(f"detect_schema: {path.name} -> {len(df)} rows, {len(df.columns)} columns, cleaning={cleaning_report}")

    columns = [_infer_column(df[col]) for col in df.columns]
    result = {
        "file_name": path.name,
        "sheet": sheet,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "data_quality": cleaning_report,
    }
    logger.info(f"detect_schema: done, types={[c['inferred_type'] for c in columns]}")
    return result


def import_dataset(path: str | Path, dataset_name: str, sheet: str | int | None = None,
                   source_label: str | None = None) -> dict:
    """Load a CSV/Excel file into its own table in user_data.duckdb.

    Column names are NOT sanitized or renamed: DuckDB references them
    through a registered DataFrame view rather than hand-built DDL, so
    arbitrary characters in a header do not create a SQL-injection
    surface the way they would if this concatenated a CREATE TABLE
    statement by hand. `dataset_name` (chosen by the caller, used as an
    actual table name) is the one identifier validated against a strict
    pattern for exactly that reason.

    source_label overrides the file name recorded and returned as
    file_name/source_file. Needed because the web upload endpoint saves
    the file under a random id (two uploads named the same thing must not
    collide on disk) - without this, every uploaded dataset would show
    the caller its meaningless server-side storage name instead of the
    name the user actually uploaded.
    """
    path = Path(path)
    dataset_name = _validate_dataset_name(dataset_name)
    display_name = source_label or path.name
    logger.info(f"import_dataset: start path={path}, dataset_name={dataset_name!r}, sheet={sheet!r}, display_name={display_name!r}")

    schema = detect_schema(path, sheet)
    schema["file_name"] = display_name
    df = _read_table(path, sheet)
    df, _ = _clean_dataframe(df)  # same cleaning detect_schema already applied and reported in schema["data_quality"]
    if len(df) > MAX_ROWS_PER_IMPORT:
        logger.warning(f"import_dataset: {len(df)} rows exceeds cap {MAX_ROWS_PER_IMPORT}, truncating")
        df = df.head(MAX_ROWS_PER_IMPORT)
        schema["truncated_to"] = MAX_ROWS_PER_IMPORT
        # schema["row_count"] was set by detect_schema() BEFORE this
        # truncation - left stale it would claim more rows were imported
        # than actually were, which is exactly the kind of silently-wrong
        # number this project does not ship. A test caught this live.
        schema["row_count"] = len(df)

    imported_at = _write_dataset(dataset_name, display_name, df)
    return {"dataset_name": dataset_name, "imported_at": imported_at, **schema}


def _write_dataset(dataset_name: str, display_name: str, df: pd.DataFrame) -> str:
    """The actual lakehouse write, shared by import_dataset (CSV/Excel)
    and pdf_import.py (PDF table extraction) - one write path so the
    collision check, registry bookkeeping, and retry-safe connection
    handling are not duplicated between them. Caller has already
    validated dataset_name and built the final DataFrame; this only
    writes it and returns the ISO timestamp it was recorded under."""
    con = _db(read_only=False)
    try:
        _reject_curated_collision(con, dataset_name)
        con.execute("BEGIN TRANSACTION")
        con.register("_incoming_df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{dataset_name}" AS SELECT * FROM _incoming_df')
        con.unregister("_incoming_df")

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
                dataset_name VARCHAR PRIMARY KEY,
                source_file VARCHAR,
                imported_at VARCHAR,
                row_count INTEGER,
                column_count INTEGER
            )
        """)
        imported_at = datetime.now(timezone.utc).isoformat()
        con.execute(
            f"INSERT OR REPLACE INTO {REGISTRY_TABLE} VALUES (?, ?, ?, ?, ?)",
            [dataset_name, display_name, imported_at, len(df), len(df.columns)],
        )
        con.execute("COMMIT")
        logger.info(f"_write_dataset: done, {dataset_name!r} <- {len(df)} rows x {len(df.columns)} cols")
        return imported_at
    finally:
        con.close()


def list_imported_datasets() -> dict:
    """Every currently-imported dataset with its columns - the equivalent
    of lakehouse_schema, but for user uploads. Returns an empty list
    rather than raising when nothing has been imported yet, since "no
    datasets" is a normal state for the model to see and report, not a
    failure."""
    logger.info("list_imported_datasets: start")
    if not LAKEHOUSE_DB_PATH.exists():
        logger.info("list_imported_datasets: lakehouse.duckdb does not exist yet")
        return {"datasets": []}

    con = _db(read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if REGISTRY_TABLE not in tables:
            return {"datasets": []}
        rows = con.execute(
            f"SELECT dataset_name, source_file, imported_at, row_count, column_count FROM {REGISTRY_TABLE} "
            f"ORDER BY imported_at DESC"
        ).fetchall()
        datasets = []
        for name, source_file, imported_at, row_count, column_count in rows:
            columns = [r[0] for r in con.execute(f'DESCRIBE "{name}"').fetchall()] if name in tables else []
            datasets.append({
                "dataset_name": name, "source_file": source_file, "imported_at": imported_at,
                "row_count": row_count, "column_count": column_count, "columns": columns,
            })
        logger.info(f"list_imported_datasets: {len(datasets)} dataset(s)")
        return {"datasets": datasets}
    finally:
        con.close()


def query_imported_dataset(dataset_name: str, sql: str) -> dict:
    """Run one read-only SELECT against an imported dataset, through the
    same guardrail as lakehouse_query (src/agent/sql_guard.py) and now
    the same physical file - lakehouse_query can reach this table too and
    JOIN it against gold_fact_long; this tool exists for the common case
    of asking about one imported dataset on its own, with 'unknown
    dataset' framed against _import_registry rather than every table in
    the lakehouse (a curated gold_* table is not an 'imported dataset',
    even though it lives in the same file now)."""
    dataset_name = _validate_dataset_name(dataset_name)
    logger.info(f"query_imported_dataset: dataset={dataset_name!r}, sql={sql!r}")
    con = _db(read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        known = {row[0] for row in con.execute(f"SELECT dataset_name FROM {REGISTRY_TABLE}").fetchall()} if REGISTRY_TABLE in tables else set()
        if dataset_name not in known:
            logger.warning(f"query_imported_dataset: unknown dataset {dataset_name!r}, known={sorted(known)}")
            raise ValueError(f"Unknown dataset {dataset_name!r}. Imported datasets: {sorted(known)}")
        return run_guarded_select(con, sql, label="query_imported_dataset")
    finally:
        con.close()


def _existing_series_meta(con: duckdb.DuckDBPyConnection, seri_kod: str) -> dict:
    """Reuse whatever Seri_Adi/Birim/Tip a series already has in
    gold_fact_long (from the last batch build) rather than guessing them
    again - config/tuik_series.py's registry carries no unit, and a
    plausible-looking guess is exactly the kind of unconfirmed label this
    project avoids elsewhere (see build_gold.py's own unit comments)."""
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "gold_fact_long" not in tables:
        return {"Seri_Adi": None, "Birim": None, "Tip": "stock"}
    row = con.execute(
        "SELECT Seri_Adi, Birim, Tip FROM gold_fact_long WHERE Seri_Kod = ? LIMIT 1", [seri_kod]
    ).fetchone()
    if row is None:
        return {"Seri_Adi": None, "Birim": None, "Tip": "stock"}
    return {"Seri_Adi": row[0], "Birim": row[1], "Tip": row[2]}


def _persist_tuik_refresh(con: duckdb.DuckDBPyConnection, seri_kod: str, label: str, periods_values: pd.DataFrame) -> int:
    """Replace this series' rows in gold_fact_long with the freshly
    fetched ones, tagged Kaynak='tuik_live' (not 'tuik') so a row that
    came from an on-demand refresh stays distinguishable from the last
    offline batch build in the same catalog a query would see. Pulled out
    of refresh_tuik_series so it can be unit-tested against a fake
    dataframe without a live TUIK call."""
    meta = _existing_series_meta(con, seri_kod)
    frame = pd.DataFrame({
        "Period": periods_values["Period"].astype(str),
        "Bolge": "TR",
        "Kaynak": "tuik_live",
        "Seri_Kod": seri_kod,
        "Seri_Adi": meta["Seri_Adi"] or label,
        "Deger": periods_values["Value"],
        "Birim": meta["Birim"],
        "Tip": meta["Tip"] or "stock",
        "Kaynak_Dosya": f"tuik_live_refresh/{seri_kod}.csv",
    })
    con.execute("DELETE FROM gold_fact_long WHERE Seri_Kod = ?", [seri_kod])
    con.register("_tuik_refresh_df", frame)
    con.execute(
        "INSERT INTO gold_fact_long (Period, Bolge, Kaynak, Seri_Kod, Seri_Adi, Deger, Birim, Tip, Kaynak_Dosya) "
        "SELECT Period, Bolge, Kaynak, Seri_Kod, Seri_Adi, Deger, Birim, Tip, Kaynak_Dosya FROM _tuik_refresh_df"
    )
    con.unregister("_tuik_refresh_df")
    return len(frame)


def refresh_tuik_series(seri_kod: str) -> dict:
    """Re-fetch one ALREADY-REGISTERED TUIK series live and write the
    fresh values into gold_fact_long (Kaynak='tuik_live'), so the same
    lakehouse_query the model already uses can see and join them
    immediately - and so does the next question in the same session,
    without re-fetching.

    Deliberately restricted to config/tuik_series.py's registry rather
    than accepting an arbitrary dataflow/key from the model: TUIK's SDMX
    API addresses a series by a multi-dimension key that has to be
    resolved by hand against the dataflow's own codelists (see that
    file's own docstring - a real example, DF_DOGUM_KDH_C, was
    investigated and abandoned because the correct key could not be
    confirmed). A model guessing a key would either get "NoRecordsFound"
    or, worse, a real but WRONG series silently labelled as the one that
    was asked for. Refreshing a known-good key is safe; inventing a new
    one is not, so this tool only does the former.

    Reuses the existing collector/processor rather than a second TUIK
    client: one code path that talks to TUIK, not two that can drift.
    """
    logger.info(f"refresh_tuik_series: start seri_kod={seri_kod!r}")
    registered = {s["code"]: s for s in TUIK_REGISTERED_SERIES}
    if seri_kod not in registered:
        logger.warning(f"refresh_tuik_series: {seri_kod!r} not registered, known={sorted(registered)}")
        raise ValueError(
            f"'{seri_kod}' is not a registered TUIK series. Registered: {sorted(registered)}. "
            f"An unregistered TUIK series cannot be fetched safely - see config/tuik_series.py."
        )

    from src.collectors.tuik import fetch_series
    from src.processing.tuik_silver import get_silver_path, parse_series

    spec = registered[seri_kod]
    ok = fetch_series(spec["dataflow_id"], spec["version"], spec["key"], seri_kod, force=True)
    if not ok:
        logger.error(f"refresh_tuik_series: live fetch failed for {seri_kod!r}")
        raise RuntimeError(f"Could not fetch '{seri_kod}' from TUIK right now; the last cached value is unchanged.")

    if not parse_series(seri_kod):
        logger.error(f"refresh_tuik_series: parse failed for {seri_kod!r} after a successful fetch")
        raise RuntimeError(f"Fetched '{seri_kod}' but could not parse the response.")

    df = pd.read_csv(get_silver_path(seri_kod))

    con = _db(read_only=False)
    try:
        written = _persist_tuik_refresh(con, seri_kod, spec["label"], df)
    finally:
        con.close()

    result = {
        "seri_kod": seri_kod,
        "label": spec["label"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "periods": df["Period"].astype(str).tolist(),
        "values": [None if pd.isna(v) else float(v) for v in df["Value"]],
        "note": f"Fetched live from TUIK and written into gold_fact_long ({written} row(s), Kaynak='tuik_live') - queryable immediately via lakehouse_query.",
    }
    logger.info(f"refresh_tuik_series: done, {len(result['periods'])} point(s), latest={result['periods'][-1] if result['periods'] else None}")
    return result


def search_tuik_dataflows(query: str, limit: int = 10) -> dict:
    """Search TUIK's full dataflow catalog (~430 dataflows) by keyword -
    metadata only, never fetches or guesses at actual data. Team feedback
    pushed back on refresh_tuik_series being limited to a 2-series
    registry: "her istediğini getirsin" (it should fetch whatever is
    asked). A flat rejection for anything unregistered was the safe
    answer but not a very USEFUL one - this is the safe middle ground:
    the model can now show the user real candidate dataflow_ids and
    titles instead of just refusing, without the actual risk (an
    unresolved SDMX key returning "NoRecordsFound" or, worse, a real but
    WRONG series silently mislabeled - see refresh_tuik_series' own
    docstring) that made arbitrary fetching unsafe in the first place.
    A result from this tool is a lead for a human to resolve further
    (config/tuik_series.py's own docstring shows what that process looks
    like), not something refresh_tuik_series can act on directly."""
    from src.collectors.tuik import search_dataflows

    matches = search_dataflows(query, limit=limit)
    result = {
        "query": query,
        "match_count": len(matches),
        "matches": [{"dataflow_id": m["dataflow_id"], "version": m["version"], "name_tr": m["name_tr"], "name_en": m["name_en"]} for m in matches],
        "note": (
            "Metadata search only - none of this data has been fetched. A dataflow_id here cannot be "
            "passed to refresh_tuik_series directly; that tool only works for the series already "
            "registered in config/tuik_series.py, because turning a dataflow_id into a fetchable "
            "series 'key' requires resolving its dimensions against that dataflow's own SDMX "
            "codelists by hand - guessing risks a real but wrong series being reported as correct."
        ),
    }
    logger.info(f"search_tuik_dataflows: query={query!r} -> {len(matches)} match(es)")
    return result


def fuzzy_match_category(query: str, candidates: list[str], threshold: float = 0.6) -> dict:
    """Reconcile a category/sector/label name against a list of
    candidates when two sources name the same thing differently - the
    real scenario: BDDK's credit-by-sector table uses a long official
    label ('Tarım, Avcılık ve Ormancılık') while TUIK's employment-by-
    sector table uses a short one ('Tarım') for the same sector, with no
    shared code to JOIN on. Get `candidates` from the OTHER source first
    (e.g. `SELECT DISTINCT Sektor FROM ...` via lakehouse_query or
    query_imported_dataset), then call this to find which one `query`
    actually refers to before writing a JOIN or a WHERE clause - do not
    guess a spelling variant yourself.

    Returns match=None (not a forced guess) when nothing clears
    `threshold` - say so plainly rather than picking the least-bad
    candidate."""
    from src.agent.data_cleaning import fuzzy_best_match

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list of strings.")
    result = fuzzy_best_match(query, candidates, threshold=threshold)
    logger.info(f"fuzzy_match_category: query={query!r}, {len(candidates)} candidate(s) -> {result}")
    if result is None:
        return {"query": query, "match": None, "score": None, "note": f"No candidate scored above threshold={threshold}."}
    name, score = result
    return {"query": query, "match": name, "score": score}
