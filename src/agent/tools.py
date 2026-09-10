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

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
logger = get_logger("causality_tool")


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


def _align_series(x_periods: list[str], x_values: list[float], y_periods: list[str], y_values: list[float]) -> pd.DataFrame:
    """Shared helper: joins two named series on Period, drops rows missing either side."""
    logger.debug(f"_align_series input: len(x)={len(x_values)}, len(y)={len(y_values)}")
    left = pd.DataFrame({"period": x_periods, "x": x_values})
    right = pd.DataFrame({"period": y_periods, "y": y_values})
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
    logger.info(f"causality_analysis done: {result}")
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
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables and columns in the local DuckDB gold layer.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a read-only SELECT query against the local DuckDB gold layer. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in an ordered time series.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}, "threshold": {"type": "number", "default": 3.0}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Calculate aligned Pearson correlation between two series; never call it causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "causality_analysis", "description": "Test whether series x is a statistically useful predictor of series y: best-aligned lag correlation plus a Granger causality test (p-values per lag). Not proof of true causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}, "max_lag": {"type": "integer", "default": 6}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "threshold_filter_compare", "description": "Filter periods where filter_values falls in [low, high] (e.g. an interest rate range), then compare the average period-over-period change of compare_values inside vs outside that window. Matches the hackathon demo's 'faiz X ile Y arasindayken kredi nasil degisti' pattern.", "parameters": {"type": "object", "properties": {"filter_periods": {"type": "array", "items": {"type": "string"}}, "filter_values": {"type": "array", "items": {"type": "number"}}, "compare_periods": {"type": "array", "items": {"type": "string"}}, "compare_values": {"type": "array", "items": {"type": "number"}}, "low": {"type": "number"}, "high": {"type": "number"}}, "required": ["filter_periods", "filter_values", "compare_periods", "compare_values", "low", "high"], "additionalProperties": False}}},
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
