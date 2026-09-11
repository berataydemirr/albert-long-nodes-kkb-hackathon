"""Local tools exposed to the MIA agent.

The tools are deliberately network-free. The model plans; Python executes
queries and calculations against the local DuckDB gold layer.
"""

import json
import math
import re
import duckdb
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config.settings import GOLD_DIR
from src.agent.anomaly_tool import ANOMALY_TOOL_SCHEMA, anomaly_analysis  # C4
from src.agent.change_detection_tool import CHANGE_DETECTION_TOOL_SCHEMA, change_detection  # C6
from src.common.logging_setup import get_logger

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
logger = get_logger("agent_tools")


def _quarterly_sort_key(period: str) -> tuple[int, int]:
    """'YYYY-Q' (Q in 3,6,9,12, the FinTurk regional table's own format)
    -> (year, quarter) for correct chronological sorting/max.

    Bug found live: plain string comparison / SQL MAX(Period) is WRONG
    here because these periods aren't zero-padded - '2021-12' < '2021-9'
    lexicographically (comparing '1' vs '9' at the first differing
    char), even though Q4 comes after Q3. Confirmed: SELECT MAX(Period)
    FROM gold_finturk_regional_quarterly happened to return the correct
    latest period today only because the most recent year in the data
    doesn't have a Q4 yet - it will silently return the wrong 'latest
    quarter' (Q3 instead of Q4) the moment one is added. Every 'latest
    period' lookup against this table must sort with this key, not
    Python's default string ordering or SQL's MAX()."""
    year_str, q_str = period.split("-")
    return (int(year_str), int(q_str))

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
# Single-quoted SQL string literal, including the standard '' escaped-quote
# form (e.g. 'O''Brien') and a backslash-escape form some dialects accept.
# Used to blank out literal content before the forbidden-token/semicolon/
# table-name checks below, so a harmless value like 'please insert coin'
# or 'a;b' can't false-positive as DML or a second statement (confirmed
# live: both were rejected before this existed).
_SQL_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.|'')*'", re.DOTALL)


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

    Guardrail, in order (checked against the SQL with comments AND string
    literal contents blanked out first - a '--'/'/* */' comment can't
    execute and neither can the text inside a 'quoted string', so neither
    must be able to either trigger a false rejection by containing a
    dangerous-looking word, or hide a real second statement from the
    semicolon check. Confirmed live: SELECT 'please insert coin' AS note
    and SELECT 'a;b' AS x were both false-rejected before literals were
    stripped here too, same as an unstripped comment would have done):
      1. single statement only (no ';' other than an optional trailing one)
      2. must start with SELECT or WITH (a CTE), no DDL/DML keywords anywhere
      3. every referenced table must exist in the gold database (SHOW TABLES)
      4. result is HARD-capped at MAX_ROWS regardless of what LIMIT (if
         any) the query itself specifies - a resource/context-size
         guarantee, not just a default for a query that forgot one
    Every rejection is logged with the reason and the original SQL, so a
    bad query from the model is traceable instead of a silent 'it just
    didn't answer'. The ORIGINAL sql (comments and literals intact) is
    what actually executes - only the guardrail checks see the blanked
    version."""
    logger.info(f"lakehouse_query: received sql={sql!r}")
    if not isinstance(sql, str):
        logger.warning(f"lakehouse_query: rejected (sql is not a string): {sql!r}")
        raise ValueError("sql must be a string.")

    stripped_of_comments = _SQL_COMMENT_RE.sub(" ", sql)
    stripped_of_literals = _SQL_STRING_LITERAL_RE.sub("''", stripped_of_comments)
    normalized = re.sub(r"\s+", " ", stripped_of_literals.strip()).lower()

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
        # DuckDB folds unquoted identifiers case-insensitively (FROM
        # GOLD_FACT_LONG executes fine against a catalog table named
        # gold_fact_long) - comparing raw case here would reject a query
        # DuckDB itself would happily run. Lowercase both sides for the
        # membership check; report the model's own original casing back
        # in the error so it's still traceable.
        known_tables_lower = {t.lower() for t in known_tables}
        # Also run against the literal-stripped SQL, not just
        # comment-stripped - otherwise a string like 'select * from
        # users' embedded as a harmless literal value spuriously matches
        # the FROM-name regex and gets rejected as an unknown table.
        cte_names_lower = {n.lower() for n in _cte_names(stripped_of_literals)}
        referenced = _referenced_tables(stripped_of_literals)
        unknown = {r for r in referenced if r.lower() not in known_tables_lower and r.lower() not in cte_names_lower}
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
            # statsmodels returns dict keys as numpy.int64, not a plain
            # int - json.dumps(default=str) upstream was silently
            # stringifying it ("2" instead of 2), and a stricter JSON
            # encoder (FastAPI's) refuses it outright. Cast once, here,
            # so every consumer downstream sees a real int.
            p_value = float(res[0]["ssr_ftest"][1])
            granger_pvalues.append({"lag": int(lag), "p_value": round(p_value, 4)})
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

    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        logger.error(f"threshold_filter_compare: low/high must be numbers, got low={low!r} high={high!r}")
        raise ValueError(f"low and high must both be numbers, got low={low!r} ({type(low).__name__}), high={high!r} ({type(high).__name__})")

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


def describe_series(periods: list[str], values: list[float]) -> dict:
    """Summary statistics for one series: range, spread, and total change
    from first to last OBSERVED (non-missing) point - not a substitute
    for anomaly_analysis/change_detection, but a cheap first look that
    doesn't need a second lakehouse_query round-trip just to answer
    'what's the range of this series'."""
    logger.info(f"describe_series start: n={len(values)}")
    if len(periods) != len(values):
        logger.error(f"describe_series: length mismatch periods={len(periods)} values={len(values)}")
        raise ValueError(f"periods and values must have equal length, got {len(periods)} and {len(values)}")
    if len(periods) == 0:
        logger.error("describe_series: empty input")
        raise ValueError("At least 1 point is required")

    def _is_missing(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    # Input order is treated as chronological, same assumption every
    # other tool in this file makes (change_detection, anomaly_analysis)
    # - periods are NOT re-sorted here, since some gold tables use
    # non-zero-padded period strings ('YYYY-Q') where a naive string
    # sort would silently misorder Q4 before Q3 (see _quarterly_sort_key
    # above) and this function has no way to know which format it was
    # handed.
    clean = [(p, v) for p, v in zip(periods, values) if not _is_missing(v)]
    missing_count = len(values) - len(clean)
    logger.debug(f"describe_series: clean_n={len(clean)}, missing_count={missing_count}")

    if not clean:
        logger.warning("describe_series: every value is missing/NaN")
        result = {
            "count": len(values), "missing_count": missing_count,
            "first_period": None, "last_period": None,
            "min": None, "max": None, "mean": None, "median": None, "std": None,
            "first_value": None, "last_value": None,
            "total_change": None, "total_percent_change": None,
        }
        logger.info(f"describe_series done (all missing): {result}")
        return result

    clean_values = pd.Series([v for _, v in clean], dtype="float64")
    first_period, first_value = clean[0]
    last_period, last_value = clean[-1]
    total_change = float(last_value) - float(first_value)
    total_percent_change = (total_change / abs(first_value) * 100) if first_value != 0 else None

    result = {
        "count": len(values),
        "missing_count": missing_count,
        "first_period": first_period,
        "last_period": last_period,
        "min": float(clean_values.min()),
        "max": float(clean_values.max()),
        "mean": float(clean_values.mean()),
        "median": float(clean_values.median()),
        "std": float(clean_values.std()) if len(clean_values) > 1 else 0.0,
        "first_value": float(first_value),
        "last_value": float(last_value),
        "total_change": total_change,
        "total_percent_change": float(total_percent_change) if total_percent_change is not None else None,
    }
    logger.info(f"describe_series done: count={result['count']}, missing={missing_count}, range=[{result['min']},{result['max']}]")
    return result


def compare_regions(period: str | None = None, top_n: int = 10, compare_period: str | None = None) -> dict:
    """Rank provinces (Bolge) by total cash loans for one FinTurk quarter
    (gold_finturk_regional_quarterly). Defaults to the latest available
    quarter if period is omitted. If compare_period is also given,
    additionally returns each province's growth% between compare_period
    and period, sorted fastest-growing first - purpose-built so a
    regional comparison question doesn't need the agent to hand-write a
    multi-step SQL join every time."""
    logger.info(f"compare_regions start: period={period}, top_n={top_n}, compare_period={compare_period}")
    if not isinstance(top_n, int) or top_n < 1:
        logger.error(f"compare_regions: invalid top_n={top_n!r}")
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")

    con = _db()
    try:
        table = "gold_finturk_regional_quarterly"
        if table not in {row[0] for row in con.execute("SHOW TABLES").fetchall()}:
            logger.error(f"compare_regions: {table} missing from gold layer")
            raise ValueError(f"{table} table not found in the gold layer.")

        available_periods = [row[0] for row in con.execute(f'SELECT DISTINCT Period FROM "{table}"').fetchall()]
        if not available_periods:
            logger.error(f"compare_regions: {table} has no rows")
            raise ValueError(f"{table} has no data.")
        available_sorted = sorted(available_periods, key=_quarterly_sort_key)

        if period is None:
            period = available_sorted[-1]
            logger.debug(f"compare_regions: defaulted period to latest available={period}")
        elif period not in available_periods:
            logger.warning(f"compare_regions: unknown period={period!r}")
            raise ValueError(f"Unknown period {period!r}. Available periods: {available_sorted}")

        ranking_df = con.execute(
            f'SELECT Bolge, Toplam_Nakdi_Kredi_Bin_TL, Takipteki_Alacak_Bin_TL, NPL_Orani_Yaklasik '
            f'FROM "{table}" WHERE Period = ? ORDER BY Toplam_Nakdi_Kredi_Bin_TL DESC LIMIT ?',
            [period, top_n],
        ).fetchdf()
        result = {
            "period": period,
            "top_n": top_n,
            "ranking": ranking_df.where(ranking_df.notna(), None).to_dict("records"),
        }

        if compare_period is not None:
            if compare_period not in available_periods:
                logger.warning(f"compare_regions: unknown compare_period={compare_period!r}")
                raise ValueError(f"Unknown compare_period {compare_period!r}. Available periods: {available_sorted}")
            if compare_period == period:
                logger.warning("compare_regions: compare_period equals period")
                raise ValueError("compare_period must differ from period.")

            growth_df = con.execute(
                f'SELECT a.Bolge, a."Toplam_Nakdi_Kredi_Bin_TL" AS value_at_compare_period, '
                f'b."Toplam_Nakdi_Kredi_Bin_TL" AS value_at_period '
                f'FROM "{table}" a JOIN "{table}" b ON a.Bolge = b.Bolge '
                f'WHERE a.Period = ? AND b.Period = ?',
                [compare_period, period],
            ).fetchdf()
            # A province whose baseline value is 0 has an undefined
            # growth% (same reasoning as change_detection's from-zero
            # case) - excluded from the ranked list rather than
            # crashing or reporting an infinite/NaN percentage.
            growth_df = growth_df[growth_df["value_at_compare_period"] != 0].copy()
            growth_df["growth_percent"] = (
                (growth_df["value_at_period"] - growth_df["value_at_compare_period"])
                / growth_df["value_at_compare_period"].abs() * 100
            )
            growth_df = growth_df.sort_values("growth_percent", ascending=False)
            result["compare_period"] = compare_period
            result["growth"] = growth_df.where(growth_df.notna(), None).to_dict("records")

        logger.info(f"compare_regions done: period={period}, ranking_n={len(result['ranking'])}")
        return result
    finally:
        con.close()


TOOL_FUNCTIONS = {
    "lakehouse_schema": lakehouse_schema,
    "lakehouse_query": lakehouse_query,
    "anomaly_analysis": anomaly_analysis,
    "change_detection": change_detection,
    "correlation_analysis": correlation_analysis,
    "causality_analysis": causality_analysis,
    "threshold_filter_compare": threshold_filter_compare,
    "describe_series": describe_series,
    "compare_regions": compare_regions,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables/columns and (for gold_fact_long) the real Seri_Kod/Seri_Adi/Kaynak/Birim catalog in the local DuckDB gold layer. Call this before writing SQL - Seri_Kod values must come from here, never guessed.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a single read-only SELECT query against the local DuckDB gold layer. Only tables returned by lakehouse_schema may be referenced. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    ANOMALY_TOOL_SCHEMA,
    CHANGE_DETECTION_TOOL_SCHEMA,
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Calculate aligned Pearson correlation between two series; never call it causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "causality_analysis", "description": "Test whether series x is a statistically useful predictor of series y: best-aligned lag correlation plus a Granger causality test (p-values per lag). Not proof of true causation.", "parameters": {"type": "object", "properties": {"x_periods": {"type": "array", "items": {"type": "string"}}, "x_values": {"type": "array", "items": {"type": "number"}}, "y_periods": {"type": "array", "items": {"type": "string"}}, "y_values": {"type": "array", "items": {"type": "number"}}, "max_lag": {"type": "integer", "default": 6}}, "required": ["x_periods", "x_values", "y_periods", "y_values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "threshold_filter_compare", "description": "Filter periods where filter_values falls in [low, high] (e.g. an interest rate range), then compare the average period-over-period change of compare_values inside vs outside that window. Matches the hackathon demo's 'faiz X ile Y arasindayken kredi nasil degisti' pattern.", "parameters": {"type": "object", "properties": {"filter_periods": {"type": "array", "items": {"type": "string"}}, "filter_values": {"type": "array", "items": {"type": "number"}}, "compare_periods": {"type": "array", "items": {"type": "string"}}, "compare_values": {"type": "array", "items": {"type": "number"}}, "low": {"type": "number"}, "high": {"type": "number"}}, "required": ["filter_periods", "filter_values", "compare_periods", "compare_values", "low", "high"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "describe_series", "description": "Summary statistics for one series: count, missing count, min/max/mean/median/std, and total change from first to last observed point. A cheap first look before reaching for anomaly_analysis or change_detection.", "parameters": {"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "number"}}}, "required": ["periods", "values"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_regions", "description": "Rank Turkish provinces by total cash loans for one quarter (gold_finturk_regional_quarterly), defaulting to the latest available quarter. Optionally also returns each province's growth% versus an earlier compare_period, fastest-growing first.", "parameters": {"type": "object", "properties": {"period": {"type": "string", "description": "'YYYY-Q' with Q in 3,6,9,12, e.g. '2026-6'. Omit for the latest available quarter."}, "top_n": {"type": "integer", "default": 10}, "compare_period": {"type": "string", "description": "Optional earlier 'YYYY-Q' period to compute growth% against."}}, "required": [], "additionalProperties": False}}},
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
