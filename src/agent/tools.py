"""Local tools exposed to the MIA agent.

The tools are deliberately network-free. The model plans; Python executes
queries and calculations against the local DuckDB gold layer.
"""

import json
import re
import duckdb
import pandas as pd

from config.settings import GOLD_DIR
from src.common.logging_setup import get_logger

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
logger = get_logger("lakehouse_tool")

# gold_fact_long has a Seri_Kod dimension the LLM can't guess from DESCRIBE
# alone - without this it will invent Seri_Kod values that don't exist.
SERIES_CATALOG_QUERY = """
    SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim, Tip
    FROM gold_fact_long
    ORDER BY Kaynak, Seri_Kod
"""

FORBIDDEN_TOKENS = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "copy ", "attach ", "pragma ", "call ")


def _db() -> duckdb.DuckDBPyConnection:
    logger.debug(f"_db: opening {DB_PATH} (read_only=True)")
    if not DB_PATH.exists():
        logger.error(f"_db: gold database missing at {DB_PATH}")
        raise FileNotFoundError(
            f"Gold database not found: {DB_PATH}. Run `python -m src.gold.build_gold` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


def lakehouse_schema() -> dict:
    """Return tables, columns, and (for gold_fact_long) the actual Seri_Kod
    catalog - a small LLM cannot write a correct WHERE Seri_Kod=... clause
    without seeing the real codes, so this is included by default rather
    than requiring a second lookup call."""
    logger.info("lakehouse_schema: start")
    con = _db()
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        logger.debug(f"lakehouse_schema: tables={[t[0] for t in tables]}")
        result = {"tables": {}}
        for (table,) in tables:
            result["tables"][table] = [row[0] for row in con.execute(f'DESCRIBE "{table}"').fetchall()]

        if "gold_fact_long" in result["tables"]:
            catalog = con.execute(SERIES_CATALOG_QUERY).fetchdf()
            result["gold_fact_long_series_catalog"] = catalog.where(catalog.notna(), None).to_dict("records")
            logger.debug(f"lakehouse_schema: series_catalog_size={len(catalog)}")

        logger.info(f"lakehouse_schema: done, {len(result['tables'])} tables")
        return result
    finally:
        con.close()


def _referenced_tables(sql: str) -> set[str]:
    """Best-effort extraction of table names after FROM/JOIN, for the
    known-table guardrail below - not a full SQL parser, just enough to
    catch a query aimed at a table that was never in SHOW TABLES."""
    return {m.group(1).strip('"') for m in re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_\"]*)", sql, re.IGNORECASE)}


def lakehouse_query(sql: str) -> dict:
    """Execute one read-only SELECT query and return rows plus metadata.

    Guardrail, in order:
      1. single statement only (no ';' other than an optional trailing one)
      2. must start with SELECT, no DDL/DML keywords anywhere
      3. every referenced table must exist in the gold database (SHOW TABLES)
      4. LIMIT 200 added if the query doesn't specify one
    Every rejection is logged with the reason and the original SQL, so a
    bad query from the model is traceable instead of a silent 'it just
    didn't answer'."""
    logger.info(f"lakehouse_query: received sql={sql!r}")
    normalized = re.sub(r"\s+", " ", sql.strip()).lower()

    inner_sql = normalized[:-1] if normalized.endswith(";") else normalized
    if ";" in inner_sql:
        logger.warning(f"lakehouse_query: rejected (multiple statements): {sql!r}")
        raise ValueError("Lakehouse tool only permits a single SELECT statement (no ';' inside the query).")

    if not normalized.startswith("select") or any(token in normalized for token in FORBIDDEN_TOKENS):
        logger.warning(f"lakehouse_query: rejected (not a plain SELECT): {sql!r}")
        raise ValueError("Lakehouse tool only permits read-only SELECT queries.")

    con = _db()
    try:
        known_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        referenced = _referenced_tables(sql)
        unknown = referenced - known_tables
        if unknown:
            logger.warning(f"lakehouse_query: rejected (unknown table(s) {unknown}): {sql!r}")
            raise ValueError(f"Unknown table(s) referenced: {sorted(unknown)}. Known tables: {sorted(known_tables)}")

        if " limit " not in f" {normalized} ":
            sql = f"{sql.rstrip(';')} LIMIT 200"
            logger.debug(f"lakehouse_query: no LIMIT given, appended -> {sql!r}")

        frame = con.execute(sql).fetchdf()
        logger.info(f"lakehouse_query: success, row_count={len(frame)}, columns={list(frame.columns)}")
        return {"columns": list(frame.columns), "rows": frame.where(frame.notna(), None).to_dict("records"), "row_count": len(frame)}
    except Exception:
        logger.exception(f"lakehouse_query: execution failed for sql={sql!r}")
        raise
    finally:
        con.close()


def anomaly_analysis(periods: list[str], values: list[float], threshold: float = 3.0) -> dict:
    """Detect robust outliers using median absolute deviation."""
    if len(periods) != len(values) or len(values) < 3:
        raise ValueError("periods and values must have equal length and at least 3 points")
    series = pd.Series(values, dtype="float64")
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad == 0:
        scores = pd.Series([0.0] * len(series))
    else:
        scores = 0.6745 * (series - median).abs() / mad
    anomalies = [
        {"period": period, "value": float(value), "score": round(float(score), 3)}
        for period, value, score in zip(periods, values, scores)
        if score >= threshold
    ]
    return {"method": "median_absolute_deviation", "threshold": threshold, "anomalies": anomalies}


def change_detection(periods: list[str], values: list[float]) -> dict:
    """Return the largest period-over-period changes."""
    if len(periods) != len(values) or len(values) < 2:
        raise ValueError("periods and values must have equal length and at least 2 points")
    changes = []
    for previous, current, period in zip(values, values[1:], periods[1:]):
        if previous == 0:
            continue
        changes.append({"period": period, "absolute_change": current - previous, "percent_change": (current - previous) / abs(previous) * 100})
    changes.sort(key=lambda item: abs(item["percent_change"]), reverse=True)
    return {"largest_changes": changes[:10]}


def correlation_analysis(x_periods: list[str], x_values: list[float], y_periods: list[str], y_values: list[float]) -> dict:
    """Measure aligned Pearson correlation; explicitly not a causal claim."""
    left = pd.DataFrame({"period": x_periods, "x": x_values})
    right = pd.DataFrame({"period": y_periods, "y": y_values})
    joined = left.merge(right, on="period").dropna()
    if len(joined) < 3:
        raise ValueError("At least 3 shared periods are required")
    return {"shared_periods": len(joined), "pearson_correlation": float(joined["x"].corr(joined["y"])), "warning": "Correlation is not proof of causality."}


TOOL_FUNCTIONS = {
    "lakehouse_schema": lakehouse_schema,
    "lakehouse_query": lakehouse_query,
    "anomaly_analysis": anomaly_analysis,
    "change_detection": change_detection,
    "correlation_analysis": correlation_analysis,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables/columns and (for gold_fact_long) the real Seri_Kod/Seri_Adi/Kaynak/Birim catalog in the local DuckDB gold layer. Call this before writing SQL - Seri_Kod values must come from here, never guessed.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a single read-only SELECT query against the local DuckDB gold layer. Only tables returned by lakehouse_schema may be referenced. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in an ordered time series.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}, "threshold": {"type": "number", "default": 3.0}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Calculate aligned Pearson correlation between two series; never call it causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
]


def execute_tool(name: str, arguments: str) -> str:
    logger.info(f"execute_tool call: name={name}, arguments={arguments}")
    if name not in TOOL_FUNCTIONS:
        logger.error(f"execute_tool: unknown tool requested: {name}")
        raise ValueError(f"Unknown tool: {name}")
    try:
        result = TOOL_FUNCTIONS[name](**json.loads(arguments))
    except Exception:
        logger.exception(f"execute_tool: '{name}' raised")
        raise
    logger.debug(f"execute_tool result: name={name}, result={result}")
    return json.dumps(result, ensure_ascii=False, default=str)
