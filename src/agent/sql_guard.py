"""Read-only SELECT guardrail, shared by every DuckDB file the agent can
query (the gold lakehouse, and now the user-uploaded dataset store).

Factored out of tools.py's original lakehouse_query rather than
duplicated: two independent copies of a SQL-injection defense drift
apart over time, and the whole point of a guardrail is that every query
path goes through the exact same one.
"""

import re

import duckdb

from src.common.logging_setup import get_logger

logger = get_logger("sql_guard")

FORBIDDEN_TOKENS = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "copy ", "attach ", "pragma ", "call ")
MAX_ROWS = 200

_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
# A single-quoted SQL string literal, '' being the standard escape for a
# literal quote inside one. Blanked out (not removed - length matters for
# offsets) before the keyword/prefix checks below, same treatment
# comments already get: without this, quoted DATA containing a
# DML-shaped word ("please insert coin", a person's name with "update" in
# it, ...) is indistinguishable from a real DROP/DELETE/etc. statement and
# gets rejected as one - found live via a test with exactly that string.
_SQL_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")


# EXTRACT(YEAR FROM Period_Date) uses FROM as part of the function's own
# syntax. Left alone, the table scan below read "Period_Date" as a table
# name and rejected the query - found live: an EXTRACT(YEAR FROM ...)
# aggregate was refused with "Unknown table(s) referenced: ['Period_Date']"
# and the model wrongly concluded the column itself was missing.
# Only EXTRACT's OWN keyword is removed: a subquery nested inside the call
# keeps its FROM, so EXTRACT(year FROM (SELECT * FROM read_csv(...))) is
# still caught as an unknown table - that is the file-read path this
# guard exists to close.
_EXTRACT_FROM_RE = re.compile(r"\bextract\s*\(\s*([a-z_]+)\s+from\b", re.IGNORECASE)


def referenced_tables(sql: str) -> set[str]:
    """Best-effort extraction of table names after FROM/JOIN, for the
    known-table guardrail below - not a full SQL parser, just enough to
    catch a query aimed at a table that was never in SHOW TABLES."""
    sql = _EXTRACT_FROM_RE.sub(r"extract(\1 ", sql)
    return {m.group(1).strip('"') for m in re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_\"]*)", sql, re.IGNORECASE)}


def cte_names(sql: str) -> set[str]:
    """Names a 'WITH x AS (...), y AS (...)' clause defines - these are
    NOT real database tables, so referenced_tables() correctly finds them
    after a later FROM/JOIN, but the known-table guardrail must not
    reject them as unknown just because they're not in SHOW TABLES."""
    names = set()
    with_match = re.search(r"^\s*with\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE)
    if with_match:
        names.add(with_match.group(1))
    names.update(m.group(1) for m in re.finditer(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE))
    return names


def run_guarded_select(con: duckdb.DuckDBPyConnection, sql: str, *, label: str = "sql_guard",
                       max_rows: int = MAX_ROWS) -> dict:
    """Execute one read-only SELECT against an already-open connection.

    Guardrail, in order (checked against the SQL with comments stripped
    first - a '--'/'/* */' comment can't execute, so it must not be able
    to either trigger a false rejection by containing a dangerous-looking
    word, or hide a real second statement from the semicolon check):
      1. single statement only (no ';' other than an optional trailing one)
      2. must start with SELECT or WITH (a CTE), no DDL/DML keywords anywhere
      3. every referenced table must exist in the connection's own
         SHOW TABLES - the caller decides which database that is
      4. result is HARD-capped at max_rows regardless of what LIMIT (if
         any) the query itself specifies - a resource/context-size
         guarantee, not just a default for a query that forgot one

    Every rejection is logged with the reason and the original SQL, so a
    bad query from the model is traceable instead of a silent "it just
    didn't answer". The connection is the caller's - opened and closed by
    them, since one lakehouse call and one imported-dataset call may need
    two different files open at once.
    """
    logger.info(f"{label}: received sql={sql!r}")
    if not isinstance(sql, str):
        logger.warning(f"{label}: rejected (sql is not a string): {sql!r}")
        raise ValueError("sql must be a string.")

    # Comments AND string literals are blanked before every guardrail
    # check below - a comment can't execute, and quoted DATA is not SQL
    # syntax, so neither should be able to either trigger a false
    # rejection (a semicolon or a DML-shaped word inside quoted text) or
    # hide a real second statement. The literal `sql` passed to
    # con.execute() below is untouched - only the guardrail's own view of
    # the query is blanked.
    stripped_of_comments = _SQL_COMMENT_RE.sub(" ", sql)
    guard_text = _SQL_STRING_LITERAL_RE.sub("''", stripped_of_comments)
    normalized = re.sub(r"\s+", " ", guard_text.strip()).lower()

    inner_sql = normalized[:-1] if normalized.endswith(";") else normalized
    if ";" in inner_sql:
        logger.warning(f"{label}: rejected (multiple statements): {sql!r}")
        raise ValueError("Only a single SELECT statement is permitted (no ';' inside the query).")

    if not (normalized.startswith("select") or normalized.startswith("with")) or any(token in normalized for token in FORBIDDEN_TOKENS):
        logger.warning(f"{label}: rejected (not a plain SELECT/CTE): {sql!r}")
        raise ValueError("Only read-only SELECT queries are permitted (a leading WITH/CTE is fine).")

    known_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    referenced = referenced_tables(guard_text)
    unknown = referenced - known_tables - cte_names(guard_text)
    if unknown:
        logger.warning(f"{label}: rejected (unknown table(s) {unknown}): {sql!r}")
        raise ValueError(f"Unknown table(s) referenced: {sorted(unknown)}. Known tables: {sorted(known_tables)}")

    frame = con.execute(sql).fetchdf()
    truncated = len(frame) > max_rows
    if truncated:
        logger.warning(f"{label}: result had {len(frame)} rows, hard-capping to {max_rows}")
        frame = frame.head(max_rows)

    logger.info(f"{label}: success, row_count={len(frame)}, truncated={truncated}, columns={list(frame.columns)}")
    # frame.where(frame.notna(), None) on its own is a no-op for a float64
    # column: pandas cannot hold a Python None in a float dtype and quietly
    # coerces it straight back to np.nan. json.dumps then emits a bare NaN
    # token, which is not valid JSON - a browser's JSON.parse rejects it,
    # even though Python's own json module tolerates it both ways and so
    # never surfaced this. astype(object) first lets each cell hold an
    # arbitrary Python value, so the None actually sticks. Found live: a
    # missing value in an uploaded dataset's numeric column, not from any
    # of the vetted gold-layer series (their gaps are all forward-filled
    # before they reach here) - but the fix applies to every caller.
    safe_rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    result = {"columns": list(frame.columns), "rows": safe_rows, "row_count": len(frame)}
    if truncated:
        result["truncated"] = True
    return result
