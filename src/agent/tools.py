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
from src.agent.regional import compare_regions
from src.agent.series_resolver import resolve_series
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


def _series_input(label: str, seri_kod: str | None, periods: list[str] | None,
                  values: list[float] | None, bolge: str | None = None,
                  start_period: str | None = None, end_period: str | None = None) -> tuple[list[str], list[float], dict]:
    """Resolve one analysis input from EITHER a Seri_Kod or explicit arrays.

    seri_kod is the cheap path: the values are read from DuckDB instead of
    being retyped by the model (see src/agent/series_resolver.py for the
    measured token cost of the array path). The arrays stay supported for
    data that is not in the lakehouse at all - e.g. numbers the model just
    pulled off a web page - and because every existing caller uses them.

    Returns (periods, values, source_meta) where source_meta is echoed back
    in the tool result so the answer can cite where the numbers came from.
    """
    logger.debug(
        f"_series_input[{label}]: seri_kod={seri_kod!r}, "
        f"periods={'None' if periods is None else len(periods)}, "
        f"values={'None' if values is None else len(values)}"
    )

    if seri_kod is not None:
        if periods is not None or values is not None:
            # Ambiguous input: silently preferring one side would make the
            # result impossible to explain later, so it is rejected loudly.
            logger.error(f"_series_input[{label}]: both seri_kod and explicit arrays supplied")
            raise ValueError(
                f"Pass either {label}seri_kod or {label}periods/{label}values, not both."
            )
        logger.info(f"_series_input[{label}]: resolving from lakehouse, seri_kod={seri_kod!r}")
        resolved = resolve_series(seri_kod, bolge=bolge, start_period=start_period, end_period=end_period)
        meta = {
            "source": "lakehouse",
            "seri_kod": resolved["seri_kod"],
            "seri_adi": resolved["seri_adi"],
            "birim": resolved["birim"],
            "tip": resolved["tip"],
            "bolge": resolved["bolge"],
            "row_count": resolved["row_count"],
            "first_period": resolved["periods"][0] if resolved["periods"] else None,
            "last_period": resolved["periods"][-1] if resolved["periods"] else None,
        }
        logger.info(f"_series_input[{label}]: resolved {meta['row_count']} point(s) from {meta['first_period']} to {meta['last_period']}")
        return resolved["periods"], resolved["values"], meta

    if periods is None or values is None:
        logger.error(f"_series_input[{label}]: neither seri_kod nor complete arrays supplied")
        raise ValueError(
            f"Provide {label}seri_kod (preferred - reads the series straight from the "
            f"lakehouse) or both {label}periods and {label}values."
        )

    logger.debug(f"_series_input[{label}]: using caller-supplied arrays, n={len(values)}")
    return periods, values, {"source": "inline_arrays", "row_count": len(values)}


def anomaly_analysis(periods: list[str] | None = None, values: list[float] | None = None,
                     threshold: float = 3.0, seri_kod: str | None = None,
                     bolge: str | None = None, start_period: str | None = None,
                     end_period: str | None = None) -> dict:
    """Detect robust outliers using median absolute deviation."""
    periods, values, source = _series_input(
        "", seri_kod, periods, values, bolge=bolge, start_period=start_period, end_period=end_period
    )
    logger.info(f"anomaly_analysis start: n={len(values)}, threshold={threshold}, source={source['source']}")
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
    return {"method": "median_absolute_deviation", "threshold": threshold, "anomalies": anomalies, "source": source}


def change_detection(periods: list[str] | None = None, values: list[float] | None = None,
                     seri_kod: str | None = None, bolge: str | None = None,
                     start_period: str | None = None, end_period: str | None = None) -> dict:
    """Return the largest period-over-period changes."""
    periods, values, source = _series_input(
        "", seri_kod, periods, values, bolge=bolge, start_period=start_period, end_period=end_period
    )
    logger.info(f"change_detection start: n={len(values)}, source={source['source']}")
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
    return {"largest_changes": changes[:10], "source": source}


def correlation_analysis(x_periods: list[str] | None = None, x_values: list[float] | None = None,
                         y_periods: list[str] | None = None, y_values: list[float] | None = None,
                         x_seri_kod: str | None = None, y_seri_kod: str | None = None,
                         start_period: str | None = None, end_period: str | None = None) -> dict:
    """Measure aligned Pearson correlation; explicitly not a causal claim."""
    x_periods, x_values, x_source = _series_input(
        "x_", x_seri_kod, x_periods, x_values, start_period=start_period, end_period=end_period
    )
    y_periods, y_values, y_source = _series_input(
        "y_", y_seri_kod, y_periods, y_values, start_period=start_period, end_period=end_period
    )
    logger.info(f"correlation_analysis start: n_x={len(x_values)}, n_y={len(y_values)}, x_source={x_source['source']}, y_source={y_source['source']}")
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
    result = {"shared_periods": len(joined), "pearson_correlation": pearson,
              "x_source": x_source, "y_source": y_source,
              "warning": "Correlation is not proof of causality."}
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


def causality_analysis(x_periods: list[str] | None = None, x_values: list[float] | None = None,
                       y_periods: list[str] | None = None, y_values: list[float] | None = None,
                       max_lag: int = 6, x_seri_kod: str | None = None, y_seri_kod: str | None = None,
                       start_period: str | None = None, end_period: str | None = None) -> dict:
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
    x_periods, x_values, x_source = _series_input(
        "x_", x_seri_kod, x_periods, x_values, start_period=start_period, end_period=end_period
    )
    y_periods, y_values, y_source = _series_input(
        "y_", y_seri_kod, y_periods, y_values, start_period=start_period, end_period=end_period
    )
    logger.info(
        f"causality_analysis start: n_x={len(x_values)}, n_y={len(y_values)}, max_lag={max_lag}, "
        f"x_source={x_source['source']}, y_source={y_source['source']}"
    )

    joined = _align_series(x_periods, x_values, y_periods, y_values)
    if len(joined) < max_lag * 2 + 5:
        # Tell the caller the largest max_lag that WOULD fit instead of just
        # refusing: the model otherwise retries with another guessed lag,
        # burning a full round per attempt.
        usable_lag = max((len(joined) - 5) // 2, 0)
        logger.error(
            f"causality_analysis: insufficient shared periods ({len(joined)}) for max_lag={max_lag}; "
            f"largest usable max_lag={usable_lag}"
        )
        hint = f"Retry with max_lag={usable_lag}." if usable_lag >= 1 else "This pair has too few shared periods for any lag test."
        raise ValueError(
            f"Need at least {max_lag * 2 + 5} shared periods for max_lag={max_lag}, got {len(joined)}. {hint}"
        )

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
        "x_source": x_source,
        "y_source": y_source,
        "warning": "Statistical association only (lag correlation + Granger test), not proof of true causation. A significant Granger p-value means x's past improves prediction of y beyond y's own past.",
    }
    logger.info(f"causality_analysis done: shared_periods={result['shared_periods']}, best_lag={best_lag_entry}")
    return result


def threshold_filter_compare(filter_periods: list[str] | None = None, filter_values: list[float] | None = None,
                             compare_periods: list[str] | None = None, compare_values: list[float] | None = None,
                             low: float = None, high: float = None,
                             filter_seri_kod: str | None = None, compare_seri_kod: str | None = None,
                             start_period: str | None = None, end_period: str | None = None) -> dict:
    """Demo-scenario style causality check: filter periods where filter_values
    is in [low, high] (e.g. "faiz orani X ile Y arasindayken"), then compares
    the average period-over-period change of compare_values inside that
    filtered window against the baseline (all other periods)."""
    filter_periods, filter_values, filter_source = _series_input(
        "filter_", filter_seri_kod, filter_periods, filter_values, start_period=start_period, end_period=end_period
    )
    compare_periods, compare_values, compare_source = _series_input(
        "compare_", compare_seri_kod, compare_periods, compare_values, start_period=start_period, end_period=end_period
    )
    logger.info(
        f"threshold_filter_compare start: n_filter={len(filter_values)}, n_compare={len(compare_values)}, "
        f"range=[{low},{high}], filter_source={filter_source['source']}, compare_source={compare_source['source']}"
    )
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or isinstance(low, bool) or isinstance(high, bool):
        logger.error(f"threshold_filter_compare: low/high must be numeric, got low={low!r}, high={high!r}")
        raise ValueError("low and high must both be numbers.")

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
        "filter_source": filter_source,
        "compare_source": compare_source,
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
    "compare_regions": compare_regions,
    "web_search": web_search,
    "web_url_extract": web_url_extract,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables/columns and (for gold_fact_long) the real Seri_Kod/Seri_Adi/Kaynak/Birim catalog in the local DuckDB gold layer. Call this before writing SQL - Seri_Kod values must come from here, never guessed.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a single read-only SELECT query against the local DuckDB gold layer. Only tables returned by lakehouse_schema may be referenced. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in a time series. PREFERRED: pass seri_kod and the series is read straight from the lakehouse - do NOT run lakehouse_query first and retype the numbers. Pass periods/values only for data that is not in the lakehouse.", "parameters": {"type": "object", "properties": {"seri_kod": {"type": "string", "description": "Seri_Kod from lakehouse_schema. Cheapest and most accurate input."}, "start_period": {"type": "string", "description": "Optional 'YYYY-MM' lower bound."}, "end_period": {"type": "string", "description": "Optional 'YYYY-MM' upper bound."}, "bolge": {"type": "string", "description": "Optional region filter; defaults to every region."}, "periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "threshold": {"type": "number", "default": 3.0}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes. PREFERRED: pass seri_kod so the series is read from the lakehouse instead of being retyped.", "parameters": {"type": "object", "properties": {"seri_kod": {"type": "string", "description": "Seri_Kod from lakehouse_schema. Cheapest and most accurate input."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "bolge": {"type": "string"}, "periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Aligned Pearson correlation between two series; never call it causation. PREFERRED: pass x_seri_kod and y_seri_kod - both series are read from the lakehouse, no need to query and retype them.", "parameters": {"type": "object", "properties": {"x_seri_kod": {"type": "string", "description": "Seri_Kod of the first series."}, "y_seri_kod": {"type": "string", "description": "Seri_Kod of the second series."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "x_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "x_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "y_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "y_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "causality_analysis", "description": "Test whether series x is a statistically useful predictor of series y: best-aligned lag correlation plus a Granger causality test (p-values per lag). Not proof of true causation. PREFERRED: pass x_seri_kod and y_seri_kod - both series are read from the lakehouse directly, which is far cheaper and avoids transcription errors.", "parameters": {"type": "object", "properties": {"x_seri_kod": {"type": "string", "description": "Seri_Kod of the candidate predictor series."}, "y_seri_kod": {"type": "string", "description": "Seri_Kod of the target series."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "x_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "x_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "y_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "y_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "max_lag": {"type": "integer", "default": 6, "description": "Needs max_lag*2+5 shared periods; the error says which lag fits if this one does not."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "threshold_filter_compare", "description": "Filter periods where the filter series falls in [low, high] (e.g. an interest rate range), then compare the average period-over-period change of the compare series inside vs outside that window. Matches the demo's 'faiz X ile Y arasindayken kredi nasil degisti' pattern. PREFERRED: pass filter_seri_kod and compare_seri_kod.", "parameters": {"type": "object", "properties": {"filter_seri_kod": {"type": "string", "description": "Seri_Kod of the series the [low, high] window applies to."}, "compare_seri_kod": {"type": "string", "description": "Seri_Kod of the series whose change is compared."}, "low": {"type": "number"}, "high": {"type": "number"}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "filter_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "filter_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "compare_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "compare_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "required": ["low", "high"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_regions", "description": "Rank Turkish provinces by any of ~35 metrics (lending, deposits, consumer credit by product, per-capita figures, branch counts, NPL and sector default ratios) for one quarter, optionally with growth against an earlier quarter. This is the ONLY way to reach province-level data - gold_fact_long is national (Bolge='TR') and has no regional breakdown. Returns a ranked head plus totals, not all 82 rows. 'YURT DISI' is not a province and is excluded from the ranking by default.", "parameters": {"type": "object", "properties": {"period": {"type": "string", "description": "Quarter as 'YYYY-Q' with Q in 3/6/9/12, e.g. '2026-6'. Defaults to the latest available quarter."}, "metric": {"type": "string", "default": "kredi", "description": "One of ~35 province-level metrics. Common: kredi (total cash loans), npl (non-performing ratio %), mevduat (deposits), konut_kredisi, tasit_kredisi, kredi_karti, takip_konut_kredisi, kisi_basi_kredi, kisi_basi_mevduat, sube_sayisi, kredi_mevduat_orani, konut_takip_orani, insaat_takip_orani. An unknown name returns the full list."}, "top_n": {"type": "integer", "default": 10, "description": "How many provinces to return, capped at 25."}, "order": {"type": "string", "enum": ["desc", "asc"], "default": "desc"}, "compare_to_period": {"type": "string", "description": "An earlier quarter; adds growth_pct per province."}, "include_non_provinces": {"type": "boolean", "default": False, "description": "Keep 'YURT DISI' in the ranking. Leave false for a province ranking."}}, "additionalProperties": False}}},
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
