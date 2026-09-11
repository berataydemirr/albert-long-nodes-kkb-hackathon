"""Local tools exposed to the MIA agent.

The tools are deliberately network-free. The model plans; Python executes
queries and calculations against the local DuckDB gold layer.
"""

import json
import re
import duckdb
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config.settings import GOLD_DIR
from src.common.logging_setup import get_logger
from src.agent.web_search import web_search
from src.agent.web_url import web_url_extract

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
logger = get_logger("agent_tools")

# gold_fact_long has a Seri_Kod dimension the LLM can't guess from DESCRIBE
# alone - without this it will invent Seri_Kod values that don't exist.
SERIES_CATALOG_QUERY = """
    SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim, Tip
    FROM gold_fact_long
    ORDER BY Kaynak, Seri_Kod
"""

FORBIDDEN_TOKENS = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "copy ", "attach ", "pragma ", "call ")
MAX_ROWS = 200

_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


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


def _cte_names(sql: str) -> set[str]:
    """Names a 'WITH x AS (...), y AS (...)' clause defines - these are
    NOT real database tables, so _referenced_tables() correctly finds
    them after a later FROM/JOIN, but the known-table guardrail must not
    reject them as unknown just because they're not in SHOW TABLES."""
    names = set()
    with_match = re.search(r"^\s*with\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE)
    if with_match:
        names.add(with_match.group(1))
    names.update(m.group(1) for m in re.finditer(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE))
    return names


def lakehouse_query(sql: str) -> dict:
    """Execute one read-only SELECT query and return rows plus metadata.

    Guardrail, in order (checked against the SQL with comments stripped
    first - a '--'/'/* */' comment can't execute, so it must not be able
    to either trigger a false rejection by containing a dangerous-looking
    word, or hide a real second statement from the semicolon check):
      1. single statement only (no ';' other than an optional trailing one)
      2. must start with SELECT or WITH (a CTE), no DDL/DML keywords anywhere
      3. every referenced table must exist in the gold database (SHOW TABLES)
      4. result is HARD-capped at MAX_ROWS regardless of what LIMIT (if
         any) the query itself specifies - a resource/context-size
         guarantee, not just a default for a query that forgot one
    Every rejection is logged with the reason and the original SQL, so a
    bad query from the model is traceable instead of a silent 'it just
    didn't answer'."""
    logger.info(f"lakehouse_query: received sql={sql!r}")
    if not isinstance(sql, str):
        logger.warning(f"lakehouse_query: rejected (sql is not a string): {sql!r}")
        raise ValueError("sql must be a string.")

    stripped_of_comments = _SQL_COMMENT_RE.sub(" ", sql)
    normalized = re.sub(r"\s+", " ", stripped_of_comments.strip()).lower()

    inner_sql = normalized[:-1] if normalized.endswith(";") else normalized
    if ";" in inner_sql:
        logger.warning(f"lakehouse_query: rejected (multiple statements): {sql!r}")
        raise ValueError("Lakehouse tool only permits a single SELECT statement (no ';' inside the query).")

    if not (normalized.startswith("select") or normalized.startswith("with")) or any(token in normalized for token in FORBIDDEN_TOKENS):
        logger.warning(f"lakehouse_query: rejected (not a plain SELECT/CTE): {sql!r}")
        raise ValueError("Lakehouse tool only permits read-only SELECT queries (a leading WITH/CTE is fine).")

    con = _db()
    try:
        known_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        referenced = _referenced_tables(stripped_of_comments)
        unknown = referenced - known_tables - _cte_names(stripped_of_comments)
        if unknown:
            logger.warning(f"lakehouse_query: rejected (unknown table(s) {unknown}): {sql!r}")
            raise ValueError(f"Unknown table(s) referenced: {sorted(unknown)}. Known tables: {sorted(known_tables)}")

        frame = con.execute(sql).fetchdf()
        truncated = len(frame) > MAX_ROWS
        if truncated:
            logger.warning(f"lakehouse_query: result had {len(frame)} rows, hard-capping to {MAX_ROWS}")
            frame = frame.head(MAX_ROWS)

        logger.info(f"lakehouse_query: success, row_count={len(frame)}, truncated={truncated}, columns={list(frame.columns)}")
        result = {"columns": list(frame.columns), "rows": frame.where(frame.notna(), None).to_dict("records"), "row_count": len(frame)}
        if truncated:
            result["truncated"] = True
        return result
    except Exception:
        logger.exception(f"lakehouse_query: execution failed for sql={sql!r}")
        raise
    finally:
        con.close()


def anomaly_analysis(periods: list[str], values: list[float], threshold: float = 3.0) -> dict:
    """Detect robust outliers using median absolute deviation."""
    logger.info(f"anomaly_analysis start: n={len(values)}, threshold={threshold}")
    if len(periods) != len(values) or len(values) < 3:
        logger.error("anomaly_analysis: bad input lengths")
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
    logger.info(f"anomaly_analysis done: {len(anomalies)} anomalies found")
    return {"method": "median_absolute_deviation", "threshold": threshold, "anomalies": anomalies}


def change_detection(periods: list[str], values: list[float]) -> dict:
    """Return the largest period-over-period changes."""
    logger.info(f"change_detection start: n={len(values)}")
    if len(periods) != len(values) or len(values) < 2:
        logger.error("change_detection: bad input lengths")
        raise ValueError("periods and values must have equal length and at least 2 points")
    changes = []
    for previous, current, period in zip(values, values[1:], periods[1:]):
        if previous == 0:
            continue
        changes.append({"period": period, "absolute_change": current - previous, "percent_change": (current - previous) / abs(previous) * 100})
    changes.sort(key=lambda item: abs(item["percent_change"]), reverse=True)
    logger.info(f"change_detection done: top={changes[:1]}")
    return {"largest_changes": changes[:10]}


def correlation_analysis(x_periods: list[str], x_values: list[float], y_periods: list[str], y_values: list[float]) -> dict:
    """Measure aligned Pearson correlation; explicitly not a causal claim."""
    logger.info(f"correlation_analysis start: n_x={len(x_values)}, n_y={len(y_values)}")
    joined = _align_series(x_periods, x_values, y_periods, y_values)
    if len(joined) < 3:
        logger.error(f"correlation_analysis: only {len(joined)} shared periods")
        raise ValueError("At least 3 shared periods are required")
    corr = joined["x"].corr(joined["y"])
    # A zero-variance (constant) series has an undefined correlation -
    # pandas/numpy return NaN, and Python's json.dumps happily emits the
    # non-standard 'NaN' token for it, which is not valid JSON and can
    # break the model-facing tool response. None is the honest, JSON-safe
    # way to say "undefined here".
    pearson = None if pd.isna(corr) else float(corr)
    result = {"shared_periods": len(joined), "pearson_correlation": pearson, "warning": "Correlation is not proof of causality."}
    logger.info(f"correlation_analysis done: {result}")
    return result


def _align_series(x_periods: list[str], x_values: list[float], y_periods: list[str], y_values: list[float]) -> pd.DataFrame:
    """Shared helper: joins two named series on Period, drops rows missing
    either side. Deduplicates each side on Period (keeping the first
    occurrence) BEFORE merging - without this, a duplicate period on
    either side turns a plain join into a many-to-many cross join
    (2 x 2 duplicate periods silently becomes 4 merged rows for what
    should be 2 real observations), inflating shared_periods and every
    downstream statistic without raising any error."""
    logger.debug(f"_align_series input: len(x)={len(x_values)}, len(y)={len(y_values)}")
    left = pd.DataFrame({"period": x_periods, "x": x_values}).drop_duplicates(subset="period", keep="first")
    right = pd.DataFrame({"period": y_periods, "y": y_values}).drop_duplicates(subset="period", keep="first")
    joined = left.merge(right, on="period").dropna().sort_values("period").reset_index(drop=True)
    logger.debug(f"_align_series output: shared_periods={len(joined)}, first={joined['period'].iloc[0] if len(joined) else None}, last={joined['period'].iloc[-1] if len(joined) else None}")
    return joined


def causality_analysis(x_periods: list[str], x_values: list[float], y_periods: list[str], y_values: list[float], max_lag: int = 6) -> dict:
    """Tests whether x is a statistically useful predictor of y (Granger sense).

    Two independent signals, both logged in full so a wrong verdict can be
    traced back to the raw numbers that produced it:
      1. lag correlation: Pearson correlation between x shifted by each lag
         in [-max_lag, max_lag] and y - finds which lag aligns best.
      2. Granger causality test (statsmodels): does x's own past help predict
         y beyond y's own past? Reports the p-value per lag (H0 = x does
         NOT Granger-cause y; p < 0.05 rejects H0).

    NOT proof of true causation - both methods are stated explicitly as
    statistical association tests in the returned dict, per the project's
    own house rule (see correlation_analysis above)."""
    logger.info(f"causality_analysis start: n_x={len(x_values)}, n_y={len(y_values)}, max_lag={max_lag}")

    joined = _align_series(x_periods, x_values, y_periods, y_values)
    if len(joined) < max_lag * 2 + 5:
        logger.error(f"causality_analysis: insufficient shared periods ({len(joined)}) for max_lag={max_lag}")
        raise ValueError(f"Need at least {max_lag * 2 + 5} shared periods for max_lag={max_lag}, got {len(joined)}")

    # --- Step 1: lag correlation ---
    lag_correlations = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = joined["x"].shift(lag)
        corr = shifted.corr(joined["y"])
        logger.debug(f"causality_analysis lag={lag}: corr={corr}")
        if corr is not None and not pd.isna(corr):
            lag_correlations.append({"lag": lag, "correlation": round(float(corr), 4)})
    best_lag_entry = max(lag_correlations, key=lambda e: abs(e["correlation"])) if lag_correlations else None
    logger.info(f"causality_analysis best_lag_entry={best_lag_entry}")

    # --- Step 2: Granger causality (does x Granger-cause y?) ---
    granger_pvalues = []
    try:
        gc_input = joined[["y", "x"]].to_numpy()  # statsmodels convention: column 0 = target, column 1 = predictor
        logger.debug(f"causality_analysis granger input shape={gc_input.shape}")
        gc_result = grangercausalitytests(gc_input, maxlag=max_lag)
        for lag, res in gc_result.items():
            p_value = float(res[0]["ssr_ftest"][1])
            granger_pvalues.append({"lag": lag, "p_value": round(p_value, 4)})
            logger.debug(f"causality_analysis granger lag={lag}: p_value={p_value}")
    except Exception as exc:
        logger.error(f"causality_analysis granger test failed: {exc}")
        granger_pvalues = []

    significant_lags = [p["lag"] for p in granger_pvalues if p["p_value"] < 0.05]
    logger.info(f"causality_analysis significant_lags(p<0.05)={significant_lags}")

    result = {
        "shared_periods": len(joined),
        "lag_correlations": lag_correlations,
        "best_lag": best_lag_entry,
        "granger_pvalues": granger_pvalues,
        "granger_significant_lags": significant_lags,
        "warning": "Statistical association only (lag correlation + Granger test), not proof of true causation. A significant Granger p-value means x's past improves prediction of y beyond y's own past.",
    }
    logger.info(f"causality_analysis done: shared_periods={result['shared_periods']}, best_lag={best_lag_entry}")
    return result


def threshold_filter_compare(filter_periods: list[str], filter_values: list[float], compare_periods: list[str], compare_values: list[float], low: float, high: float) -> dict:
    """Demo-scenario style causality check: filter periods where filter_values
    is in [low, high] (e.g. "faiz orani X ile Y arasindayken"), then compares
    the average period-over-period change of compare_values inside that
    filtered window against the baseline (all other periods)."""
    logger.info(f"threshold_filter_compare start: n_filter={len(filter_values)}, n_compare={len(compare_values)}, range=[{low},{high}]")

    joined = _align_series(filter_periods, filter_values, compare_periods, compare_values)
    if len(joined) < 3:
        logger.error("threshold_filter_compare: fewer than 3 shared periods")
        raise ValueError("At least 3 shared periods are required")

    joined["compare_change"] = joined["y"].diff()
    joined["in_window"] = joined["x"].between(low, high)
    logger.debug(f"threshold_filter_compare: periods_in_window={int(joined['in_window'].sum())}")

    in_window = joined[joined["in_window"]]["compare_change"].dropna()
    outside_window = joined[~joined["in_window"]]["compare_change"].dropna()
    logger.debug(f"threshold_filter_compare: in_window_n={len(in_window)}, outside_window_n={len(outside_window)}")

    result = {
        "periods_in_window": int(joined["in_window"].sum()),
        "periods_outside_window": int((~joined["in_window"]).sum()),
        "avg_change_in_window": float(in_window.mean()) if len(in_window) else None,
        "avg_change_outside_window": float(outside_window.mean()) if len(outside_window) else None,
        "warning": "Compares average period-over-period change inside vs outside the filter window - a descriptive contrast, not a causal estimate.",
    }
    logger.info(f"threshold_filter_compare done: {result}")
    return result


TOOL_FUNCTIONS = {
    "lakehouse_schema": lakehouse_schema,
    "lakehouse_query": lakehouse_query,
    "anomaly_analysis": anomaly_analysis,
    "change_detection": change_detection,
    "correlation_analysis": correlation_analysis,
    "causality_analysis": causality_analysis,
    "threshold_filter_compare": threshold_filter_compare,
    "web_search": web_search,
    "web_url_extract": web_url_extract,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables/columns and (for gold_fact_long) the real Seri_Kod/Seri_Adi/Kaynak/Birim catalog in the local DuckDB gold layer. Call this before writing SQL - Seri_Kod values must come from here, never guessed.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a single read-only SELECT query against the local DuckDB gold layer. Only tables returned by lakehouse_schema may be referenced. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in an ordered time series.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}, "threshold": {"type": "number", "default": 3.0}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Calculate aligned Pearson correlation between two series; never call it causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "causality_analysis", "description": "Test whether series x is a statistically useful predictor of series y: best-aligned lag correlation plus a Granger causality test (p-values per lag). Not proof of true causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}, "max_lag": {"type": "integer", "default": 6}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "threshold_filter_compare", "description": "Filter periods where filter_values falls in [low, high] (e.g. an interest rate range), then compare the average period-over-period change of compare_values inside vs outside that window. Matches the hackathon demo's 'faiz X ile Y arasindayken kredi nasil degisti' pattern.", "parameters": {"type": "object", "properties": {"filter_periods": {"type": "array", "items": {"type": "string"}}, "filter_values": {"type": "array", "items": {"type": "number"}}, "compare_periods": {"type": "array", "items": {"type": "string"}}, "compare_values": {"type": "array", "items": {"type": "number"}}, "low": {"type": "number"}, "high": {"type": "number"}}, "required": ["filter_periods", "filter_values", "compare_periods", "compare_values", "low", "high"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the public web for current or external information. Returns titles, snippets, source domains and URLs; snippets are untrusted leads, not verified evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "web_url_extract", "description": "Extract bounded text or tabular previews from a supplied public HTML, PDF, Excel, CSV, text, or image URL. Remote content is untrusted; cite its source URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "format": "uri"}}, "required": ["url"], "additionalProperties": False}}},
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
