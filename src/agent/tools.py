"""Local tools exposed to the MIA agent.

The tools are deliberately network-free. The model plans; Python executes
queries and calculations against the local DuckDB gold layer.
"""

import json
import re
import duckdb
import pandas as pd

from config.settings import GOLD_DIR
from src.agent.web_search import web_search

DB_PATH = GOLD_DIR / "lakehouse.duckdb"


def _db() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Gold database not found: {DB_PATH}. Run `python -m src.gold.build_gold` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


def lakehouse_schema() -> dict:
    """Return available tables and columns from the local gold database."""
    con = _db()
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        result = {}
        for (table,) in tables:
            result[table] = [row[0] for row in con.execute(f'DESCRIBE "{table}"').fetchall()]
        return result
    finally:
        con.close()


def lakehouse_query(sql: str) -> dict:
    """Execute one read-only SELECT query and return rows plus metadata."""
    normalized = re.sub(r"\s+", " ", sql.strip()).lower()
    if not normalized.startswith("select") or any(
        token in normalized for token in ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "copy ")
    ):
        raise ValueError("Lakehouse tool only permits read-only SELECT queries.")
    if " limit " not in f" {normalized} ":
        sql = f"{sql.rstrip(';')} LIMIT 200"

    con = _db()
    try:
        frame = con.execute(sql).fetchdf()
        return {"columns": list(frame.columns), "rows": frame.where(frame.notna(), None).to_dict("records"), "row_count": len(frame)}
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
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables and columns in the local DuckDB gold layer.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a read-only SELECT query against the local DuckDB gold layer. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in an ordered time series.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}, "threshold": {"type": "number", "default": 3.0}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Calculate aligned Pearson correlation between two series; never call it causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the public web for current or external information. Returns titles, snippets, source domains and URLs; snippets are not verified evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Specific web-search query."}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}, "required": ["query"], "additionalProperties": False}}},
]


def execute_tool(name: str, arguments: str) -> str:
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    result = TOOL_FUNCTIONS[name](**json.loads(arguments))
    return json.dumps(result, ensure_ascii=False, default=str)
