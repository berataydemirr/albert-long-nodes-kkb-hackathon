"""Local tools exposed to the MIA agent.

The tools are deliberately network-free. The model plans; Python executes
queries and calculations against the local DuckDB gold layer.
"""

from dataclasses import dataclass
import json
import re
from typing import Callable

import duckdb
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config.settings import LAKEHOUSE_DB_PATH
from src.agent.bist_import import fetch_bist_stock
from src.agent.data_import import REGISTRY_TABLE, fuzzy_match_category, list_imported_datasets, query_imported_dataset, refresh_tuik_series, search_tuik_dataflows
from src.agent.db_utils import connect_with_retry
from src.agent.regional import compare_regions
from src.agent.series_resolver import resolve_series
from src.agent.sql_guard import run_guarded_select
from src.common.logging_setup import get_logger
from src.agent.web_search import web_search
from src.agent.web_url import web_url_extract
from src.agent.evidence import evidence_query, evidence_correlation, recall_turn
from src.agent.finance_fetch import import_url_dataset, search_evds_series, fetch_evds_series

DB_PATH = LAKEHOUSE_DB_PATH
logger = get_logger("agent_tools")

# gold_fact_long has a Seri_Kod dimension the LLM can't guess from DESCRIBE
# alone - without this it will invent Seri_Kod values that don't exist.
SERIES_CATALOG_QUERY = """
    SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim, Tip
    FROM gold_fact_long
    ORDER BY Kaynak, Seri_Kod
"""

def _db() -> duckdb.DuckDBPyConnection:
    logger.debug(f"_db: opening {DB_PATH} (read_only=True)")
    if not DB_PATH.exists():
        logger.error(f"_db: gold database missing at {DB_PATH}")
        raise FileNotFoundError(
            f"Gold database not found: {DB_PATH}. Run `python -m src.gold.build_gold` first."
        )
    return connect_with_retry(DB_PATH, read_only=True)


def _fold_tr(value: str) -> str:
    """Turkish-aware casefold for name matching (same normalization the
    EVDS/TUIK catalog searches use)."""
    return str(value).casefold().replace("i̇", "i").translate(str.maketrans("çğıöşü", "cgiosu"))


def lakehouse_schema(konu: str | None = None) -> dict:
    """Return tables, columns, and (for gold_fact_long) the actual Seri_Kod
    catalog - a small LLM cannot write a correct WHERE Seri_Kod=... clause
    without seeing the real codes, so this is included by default rather
    than requiring a second lookup call.

    konu narrows the dump to tables whose name or column names match those
    keywords. WHY it exists (measured, not assumed): the unfiltered dump is
    ~14.8k chars / ~3.7k tokens, which is over orchestrator.py's
    TOOL_OUTPUT_MAX_CHARS - so an unfiltered call was being TRUNCATED
    before the model saw it, meaning the model could be shown a schema
    with the very table it needed cut off the end, then flail and burn
    more rounds. Note the fix here is about truncation and wasted rounds,
    NOT about prompt size: a controlled A/B on this endpoint
    (benchmarks/schema_payload_ab_test.py, benchmarks/mia_latency_profile.py)
    measured 15,882 prompt tokens answering in 4.7s, i.e. input tokens are
    effectively free and wall time is dominated by generated reasoning
    tokens instead. Filtering is done with plain string matching rather
    than an embedding/RAG lookup on purpose - an embedding round-trip to
    MIA measured 2.1s, which would ADD more latency than the ~3.7k tokens
    of prefill it saves."""
    logger.info(f"lakehouse_schema: start (konu={konu!r})")
    con = _db()
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        logger.debug(f"lakehouse_schema: tables={[t[0] for t in tables]}")
        result = {"tables": {}}
        for (table,) in tables:
            result["tables"][table] = [row[0] for row in con.execute(f'DESCRIBE "{table}"').fetchall()]

        if konu and isinstance(konu, str) and konu.strip():
            needles = [word for word in _fold_tr(konu).split() if len(word) > 2]
            if needles:
                before = len(result["tables"])
                matched = {
                    name: columns for name, columns in result["tables"].items()
                    if any(n in _fold_tr(name) or any(n in _fold_tr(c) for c in columns) for n in needles)
                }
                # An over-narrow filter that matches nothing is worse than
                # no filter: the model would read "no such tables" and give
                # up. Fall back to the full list and say so.
                if matched:
                    result["tables"] = matched
                    result["filtered_by"] = konu
                    result["tables_hidden"] = before - len(matched)
                    logger.info(f"lakehouse_schema: konu={konu!r} kept {len(matched)}/{before} table(s): "
                                f"{sorted(matched)}")
                else:
                    result["filter_note"] = (
                        f"'{konu}' hiçbir tablo/kolon adıyla eşleşmedi; tam liste döndürüldü."
                    )
                    logger.warning(f"lakehouse_schema: konu={konu!r} matched no table, returning all {before}")

        if REGISTRY_TABLE in result["tables"]:
            imported = sorted(row[0] for row in con.execute(f"SELECT dataset_name FROM {REGISTRY_TABLE}").fetchall())
            if imported:
                result["user_imported_tables"] = imported
                result["user_imported_tables_note"] = (
                    "These tables were uploaded or fetched at runtime, not vetted like the gold_* "
                    "tables - state plainly that data from them is user-supplied/unverified when citing it."
                )

        if "gold_fact_long" in result["tables"]:
            catalog = con.execute(SERIES_CATALOG_QUERY).fetchdf()
            # astype(object) first: on a float64 column, .where(x.notna(),
            # None) alone is a no-op - pandas cannot hold a Python None in
            # a float dtype and silently coerces it back to np.nan, which
            # json.dumps then emits as a bare (invalid) NaN token. Found
            # live: 3 series (tablo_289.konut, M2 Para Arzi, card spending)
            # have no confirmed Birim, and every lakehouse_schema call was
            # sending the model invalid JSON for them.
            result["gold_fact_long_series_catalog"] = catalog.astype(object).where(catalog.notna(), None).to_dict("records")
            logger.debug(f"lakehouse_schema: series_catalog_size={len(catalog)}")

        logger.info(f"lakehouse_schema: done, {len(result['tables'])} tables")
        return result
    finally:
        con.close()


def lakehouse_query(sql: str) -> dict:
    """Execute one read-only SELECT query against the gold lakehouse.

    The actual guardrail (single-statement, SELECT/CTE-only, known-table
    check, row cap) lives in src/agent/sql_guard.py, shared with
    query_imported_dataset below - one guardrail, not two that can drift."""
    con = _db()
    try:
        return run_guarded_select(con, sql, label="lakehouse_query")
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
    either side. Collapses identical observations and rejects conflicting
    same-period values BEFORE merging - without this, a duplicate period on
    either side turns a plain join into a many-to-many cross join
    (2 x 2 duplicate periods silently becomes 4 merged rows for what
    should be 2 real observations), inflating shared_periods and every
    downstream statistic without raising any error."""
    logger.debug(f"_align_series input: len(x)={len(x_values)}, len(y)={len(y_values)}")
    left = pd.DataFrame({"period": x_periods, "x": x_values}).drop_duplicates()
    right = pd.DataFrame({"period": y_periods, "y": y_values}).drop_duplicates()
    for frame, column in ((left, "x"), (right, "y")):
        if frame.period.duplicated().any():
            raise ValueError("Conflicting values for the same period; select one source/version.")
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isin([float("inf"), float("-inf")]).any():
            raise ValueError("Infinite values are not valid analysis inputs.")
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


def _period_number(value: str) -> int:
    """Normalize both YYYY-M and YYYY-MM without relying on string order."""
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", str(value).strip())
    if not match:
        raise ValueError(f"Unsupported period {value!r}; expected YYYY-MM.")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"Unsupported month in period {value!r}.")
    return year * 12 + month


def credit_elasticity_analysis(
    start_period: str = "2023-01",
    end_period: str = "2024-12",
    housing_seri_kod: str = "tablo_4.konut_kredisi",
    vehicle_metric: str = "tasit_kredisi",
    policy_seri_kod: str = "TP.BISPOLFAIZ.TUR",
    cpi_seri_kod: str = "TP.GENENDEKS.T1",
) -> dict:
    """Compare CPI-deflated housing and vehicle credit sensitivity to policy rates.

    Housing, CPI and policy-rate observations are monthly national series in
    ``gold_fact_long``. Vehicle credit is a national sum of province-level,
    quarterly observations in ``gold_regional_long``. The calculation uses
    only periods shared by all inputs, so frequency alignment, inflation
    adjustment and elasticity never depend on model arithmetic.
    """
    start_no, end_no = _period_number(start_period), _period_number(end_period)
    if start_no >= end_no:
        raise ValueError("start_period must be earlier than end_period.")

    con = _db()
    try:
        fact_rows = con.execute(
            """
            SELECT Period, Seri_Kod, Deger
            FROM gold_fact_long
            WHERE Seri_Kod IN (?, ?, ?)
            ORDER BY Period
            """,
            [housing_seri_kod, policy_seri_kod, cpi_seri_kod],
        ).fetchall()
        vehicle_rows = con.execute(
            """
            SELECT Period, SUM(Deger) AS Deger
            FROM gold_regional_long
            WHERE Metrik = ?
            GROUP BY Period
            ORDER BY Period
            """,
            [vehicle_metric],
        ).fetchall()
    finally:
        con.close()

    by_code: dict[str, dict[int, tuple[str, float]]] = {
        housing_seri_kod: {}, policy_seri_kod: {}, cpi_seri_kod: {}
    }
    for period, code, value in fact_rows:
        if value is not None:
            by_code[code][_period_number(period)] = (str(period), float(value))
    vehicles = {
        _period_number(period): (str(period), float(value))
        for period, value in vehicle_rows if value is not None
    }

    missing = [code for code, values in by_code.items() if not values]
    if not vehicles:
        missing.append(f"gold_regional_long.Metrik={vehicle_metric}")
    if missing:
        raise ValueError("Required series has no observations: " + ", ".join(missing))

    shared = set(vehicles)
    for values in by_code.values():
        shared &= set(values)
    usable = sorted(period for period in shared if start_no <= period <= end_no)
    if len(usable) < 2:
        raise ValueError(
            f"Need at least two shared quarterly periods in {start_period}..{end_period}; got {len(usable)}."
        )

    first, last = usable[0], usable[-1]
    months = last - first
    if months <= 0:
        raise ValueError("Effective comparison period must span at least one month.")

    policy_start = by_code[policy_seri_kod][first][1]
    policy_end = by_code[policy_seri_kod][last][1]
    policy_values = [by_code[policy_seri_kod][period][1] for period in usable]
    cpi_start = by_code[cpi_seri_kod][first][1]
    cpi_end = by_code[cpi_seri_kod][last][1]
    if policy_start == 0 or cpi_start <= 0 or cpi_end <= 0:
        raise ValueError("Policy-rate start and CPI endpoint values must be positive.")

    rate_pp = policy_end - policy_start
    rate_relative_pct = (policy_end / policy_start - 1.0) * 100.0

    def metrics(label: str, start_value: float, end_value: float, source: dict) -> dict:
        if start_value <= 0 or end_value <= 0:
            raise ValueError(f"{label} endpoint values must be positive.")
        nominal_pct = (end_value / start_value - 1.0) * 100.0
        real_start = start_value / cpi_start
        real_end = end_value / cpi_end
        real_pct = (real_end / real_start - 1.0) * 100.0
        annualized_real_pct = ((real_end / real_start) ** (12.0 / months) - 1.0) * 100.0
        return {
            "label": label,
            "nominal_start": round(start_value, 4),
            "nominal_end": round(end_value, 4),
            "nominal_change_pct": round(nominal_pct, 4),
            "real_change_pct": round(real_pct, 4),
            "annualized_real_change_pct": round(annualized_real_pct, 4),
            "elasticity_vs_policy_rate_relative_change": (
                round(real_pct / rate_relative_pct, 6) if rate_relative_pct else None
            ),
            "semi_elasticity_real_pct_per_policy_pp": (
                round(real_pct / rate_pp, 6) if rate_pp else None
            ),
            "source": source,
        }

    housing = metrics(
        "konut_kredisi",
        by_code[housing_seri_kod][first][1],
        by_code[housing_seri_kod][last][1],
        {"table": "gold_fact_long", "seri_kod": housing_seri_kod, "frequency": "monthly"},
    )
    vehicle = metrics(
        "tasit_kredisi",
        vehicles[first][1],
        vehicles[last][1],
        {"table": "gold_regional_long", "metric": vehicle_metric,
         "aggregation": "SUM(Deger)", "frequency": "quarterly"},
    )
    more_sensitive = max(
        (housing, vehicle),
        key=lambda item: abs(item["elasticity_vs_policy_rate_relative_change"] or 0.0),
    )["label"]

    first_label = f"{first // 12:04d}-{first % 12:02d}" if first % 12 else f"{first // 12 - 1:04d}-12"
    last_label = f"{last // 12:04d}-{last % 12:02d}" if last % 12 else f"{last // 12 - 1:04d}-12"
    return {
        "requested_period": {"start": start_period, "end": end_period},
        "effective_shared_period": {"start": first_label, "end": last_label, "months": months},
        "shared_observations": len(usable),
        "policy_rate": {
            "seri_kod": policy_seri_kod,
            "start": round(policy_start, 4),
            "end": round(policy_end, 4),
            "peak_in_shared_periods": round(max(policy_values), 4),
            "change_pp": round(rate_pp, 4),
            "relative_change_pct": round(rate_relative_pct, 4),
        },
        "cpi": {"seri_kod": cpi_seri_kod, "start": round(cpi_start, 4), "end": round(cpi_end, 4)},
        "credits": [housing, vehicle],
        "more_rate_sensitive_by_absolute_elasticity": more_sensitive,
        "method": (
            "National credit stocks are compared at shared quarter-end periods. Real credit = nominal credit / CPI. "
            "Elasticity = real credit percent change / policy-rate relative percent change; semi-elasticity is "
            "real credit percent change per policy-rate percentage-point change."
        ),
        "warning": (
            "Descriptive endpoint sensitivity, not a causal estimate. Housing is a monthly national series; "
            "vehicle credit is a quarterly national sum of regional observations."
        ),
    }


_SCHEMAS = [
    {"type": "function", "function": {"name": "recall_turn", "description": "Geçmiş soru kimliğiyle tam soru, yanıt ve değişmez evidence_id kayıtlarını getir. Eski sorunun verisi bağlamda yoksa kullan; özetten sayı tahmin etme.", "parameters": {"type": "object", "properties": {"turn_id": {"type": "string"}}, "required": ["turn_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "evidence_correlation", "description": "İlk sorudaki ve sonraki sorudaki değişmez aylık veri kayıtlarını doğrula, ay bazında hizala ve DuckDB Pearson korelasyonu hesapla. Çelişkili ayları reddeder, eksik ayları raporlar. y bir TÜFE endeksiyse monthly_pct_change seç; aylık enflasyon oranıysa none. Sütun adlarını kayıt çıktısından al.", "parameters": {"type": "object", "properties": {"x_id": {"type": "string"}, "y_id": {"type": "string"}, "x_period": {"type": "string"}, "y_period": {"type": "string"}, "x_value": {"type": "string"}, "y_value": {"type": "string"}, "y_transform": {"type": "string", "enum": ["none", "monthly_pct_change"]}}, "required": ["x_id", "y_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "evidence_query", "description": "Önceki soruların değişmez sonuçlarını DuckDB ile sorgula. evidence_ids sırasıyla data_1, data_2 tablolarıdır. İlk sorudaki değerleri yeniden yazmadan JOIN ve corr hesapla. Önce tarih tekilliğini kontrol et; eksik ayları belirt.", "parameters": {"type": "object", "properties": {"evidence_ids": {"type": "array", "items": {"type": "string"}}, "sql": {"type": "string"}}, "required": ["evidence_ids", "sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "import_url_dataset", "description": "Doğrudan CSV/Excel URL'sinin tam verisini sürümlü DuckDB tablosuna al. Önizleme değildir. Sonuç dataset_name ile sorgulanır; sheet Excel sayfa adı olabilir.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "sheet": {"type": "string"}}, "required": ["url"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_evds_series", "description": "TCMB EVDS kataloğunda gerçek veri gruplarını ara; datagroup verilirse gerçek seri kodları ve metadata döner. İhtiyaç kredisini birleşik tüketici kredisiyle karıştırma. Canlı EVDS kapalıysa lakehouse'taki ve yüklenen dosyalardaki (TP_ kolonları, açıklamalarıyla) EVDS serilerini döndürür.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "datagroup": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "fetch_evds_series", "description": "Grubunda doğrulanan EVDS serisini canlı çek, ham veriyi ve aylık veriyi DuckDB'ye kaydet. Oranlar için mean, stok/endeks için last, gerçek akış için sum seç. TÜFE endeksinden aylık enflasyon türetmek için önceki ayı da çek ve SQL ile 100*(endeks/önceki_endeks-1) hesapla; eksik ay üzerinden fark alma.", "parameters": {"type": "object", "properties": {"datagroup": {"type": "string"}, "seri_kod": {"type": "string"}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "aggregation": {"type": "string", "enum": ["mean", "last", "sum"]}}, "required": ["datagroup", "seri_kod", "start_period", "end_period"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_schema", "description": "List tables/columns and (for gold_fact_long) the real Seri_Kod/Seri_Adi/Kaynak/Birim catalog in the local DuckDB gold layer. The Seri_Kod catalog is already in the system prompt, so do not call this just to find a series code - only when you need the column structure of a table the prompt does not describe. Seri_Kod values must never be guessed. ALWAYS pass 'konu' with the keywords you care about (e.g. 'kredi mevduat'): the unfiltered dump is larger than the tool-output limit and gets TRUNCATED, so the table you need may be silently cut off.", "parameters": {"type": "object", "properties": {"konu": {"type": "string", "description": "Anahtar kelimeler (ör. 'kredi', 'konut faiz', 'mevduat'). Sadece adı veya kolonları eşleşen tablolar döner; eşleşme yoksa tam liste döner."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lakehouse_query", "description": "Run a single read-only SELECT query against the local DuckDB gold layer. Only tables returned by lakehouse_schema may be referenced. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "anomaly_analysis", "description": "Find robust outliers in a time series. PREFERRED: pass seri_kod and the series is read straight from the lakehouse - do NOT run lakehouse_query first and retype the numbers. Pass periods/values only for data that is not in the lakehouse.", "parameters": {"type": "object", "properties": {"seri_kod": {"type": "string", "description": "Seri_Kod from lakehouse_schema. Cheapest and most accurate input."}, "start_period": {"type": "string", "description": "Optional 'YYYY-MM' lower bound."}, "end_period": {"type": "string", "description": "Optional 'YYYY-MM' upper bound."}, "bolge": {"type": "string", "description": "Optional region filter; defaults to every region."}, "periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "threshold": {"type": "number", "default": 3.0}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "change_detection", "description": "Find the largest period-over-period percentage changes. PREFERRED: pass seri_kod so the series is read from the lakehouse instead of being retyped.", "parameters": {"type": "object", "properties": {"seri_kod": {"type": "string", "description": "Seri_Kod from lakehouse_schema. Cheapest and most accurate input."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "bolge": {"type": "string"}, "periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "correlation_analysis", "description": "Aligned Pearson correlation between two series; never call it causation. PREFERRED: pass x_seri_kod and y_seri_kod - both series are read from the lakehouse, no need to query and retype them.", "parameters": {"type": "object", "properties": {"x_seri_kod": {"type": "string", "description": "Seri_Kod of the first series."}, "y_seri_kod": {"type": "string", "description": "Seri_Kod of the second series."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "x_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "x_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "y_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "y_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "causality_analysis", "description": "Test whether series x is a statistically useful predictor of series y: best-aligned lag correlation plus a Granger causality test (p-values per lag). Not proof of true causation. PREFERRED: pass x_seri_kod and y_seri_kod - both series are read from the lakehouse directly, which is far cheaper and avoids transcription errors.", "parameters": {"type": "object", "properties": {"x_seri_kod": {"type": "string", "description": "Seri_Kod of the candidate predictor series."}, "y_seri_kod": {"type": "string", "description": "Seri_Kod of the target series."}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "x_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "x_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "y_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "y_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "max_lag": {"type": "integer", "default": 6, "description": "Needs max_lag*2+5 shared periods; the error says which lag fits if this one does not."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "credit_elasticity_analysis", "description": "Deterministically compare CPI-deflated housing and vehicle credit contraction and sensitivity to the policy rate. Use this directly for reel daralma / faize duyarlilik / elastikiyet questions instead of separate lakehouse queries or mental arithmetic. It aligns monthly national housing/CPI/rate series with the quarterly national sum of regional vehicle credit, and returns endpoint real changes, annualized real changes, elasticity and semi-elasticity.", "parameters": {"type": "object", "properties": {"start_period": {"type": "string", "default": "2023-01"}, "end_period": {"type": "string", "default": "2024-12"}, "housing_seri_kod": {"type": "string", "default": "tablo_4.konut_kredisi"}, "vehicle_metric": {"type": "string", "default": "tasit_kredisi"}, "policy_seri_kod": {"type": "string", "default": "TP.BISPOLFAIZ.TUR"}, "cpi_seri_kod": {"type": "string", "default": "TP.GENENDEKS.T1"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "threshold_filter_compare", "description": "Filter periods where the filter series falls in [low, high] (e.g. an interest rate range), then compare the average period-over-period change of the compare series inside vs outside that window. Matches the demo's 'faiz X ile Y arasindayken kredi nasil degisti' pattern. PREFERRED: pass filter_seri_kod and compare_seri_kod.", "parameters": {"type": "object", "properties": {"filter_seri_kod": {"type": "string", "description": "Seri_Kod of the series the [low, high] window applies to."}, "compare_seri_kod": {"type": "string", "description": "Seri_Kod of the series whose change is compared."}, "low": {"type": "number"}, "high": {"type": "number"}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "filter_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "filter_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}, "compare_periods": {"type": "array", "items": {"type": "string"}, "description": "Only when the data is not in the lakehouse."}, "compare_values": {"type": "array", "items": {"type": "number"}, "description": "Only when the data is not in the lakehouse."}}, "required": ["low", "high"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_regions", "description": "Rank Turkish provinces by any of ~35 metrics (lending, deposits, consumer credit by product, per-capita figures, branch counts, NPL and sector default ratios) for one quarter, optionally with growth against an earlier quarter. This is the ONLY way to reach province-level data - gold_fact_long is national (Bolge='TR') and has no regional breakdown. Returns a ranked head plus totals, not all 82 rows. 'YURT DISI' is not a province and is excluded from the ranking by default.", "parameters": {"type": "object", "properties": {"period": {"type": "string", "description": "Quarter as 'YYYY-MM' with the month in 03/06/09/12, e.g. '2026-06'. The unpadded '2026-6' spelling is also accepted. Defaults to the latest available quarter."}, "metric": {"type": "string", "default": "kredi", "description": "One of ~35 province-level metrics. Common: kredi (total cash loans), npl (non-performing ratio %), mevduat (deposits), konut_kredisi, tasit_kredisi, kredi_karti, takip_konut_kredisi, kisi_basi_kredi, kisi_basi_mevduat, sube_sayisi, kredi_mevduat_orani, konut_takip_orani, insaat_takip_orani. An unknown name returns the full list."}, "top_n": {"type": "integer", "default": 10, "description": "How many provinces to return, capped at 25."}, "order": {"type": "string", "enum": ["desc", "asc"], "default": "desc"}, "compare_to_period": {"type": "string", "description": "An earlier quarter; adds growth_pct per province."}, "include_non_provinces": {"type": "boolean", "default": False, "description": "Keep 'YURT DISI' in the ranking. Leave false for a province ranking."}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_imported_datasets", "description": "List every dataset a user has uploaded (CSV/Excel/PDF) or the agent has fetched live (fetch_bist_stock, refresh_tuik_series), with their columns. Call this before query_imported_dataset if you are not sure a dataset exists or what it is called - lakehouse_schema also lists these tables (they live in the same database) but this gives upload/fetch provenance lakehouse_schema does not.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "query_imported_dataset", "description": "Run a single read-only SELECT against one user-supplied or live-fetched dataset (see list_imported_datasets for the name and columns). Lives in the same lakehouse as the gold_* tables and can be JOINed against them via lakehouse_query too - this tool is for the common case of asking about one such dataset on its own. This data has no confirmed source or unit the way gold_* does, so state that plainly in the answer. Results are limited to 200 rows.", "parameters": {"type": "object", "properties": {"dataset_name": {"type": "string", "description": "Exact name from list_imported_datasets."}, "sql": {"type": "string"}}, "required": ["dataset_name", "sql"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "refresh_tuik_series", "description": "Re-fetch one already-registered TUIK series live (right now, from TUIK's own API) instead of using the cached value, and write the fresh values into gold_fact_long (Kaynak='tuik_live') so they are queryable immediately via lakehouse_query. Only works for series already in the catalog (currently TUIK_UNEMPLOYMENT_RATE and TUIK_LIFE_EXPECTANCY) - an unregistered code is rejected rather than guessed, because a TUIK series key cannot be inferred safely.", "parameters": {"type": "object", "properties": {"seri_kod": {"type": "string", "description": "A Seri_Kod already present in the lakehouse catalog with Kaynak=tuik."}}, "required": ["seri_kod"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_tuik_dataflows", "description": "Search TUIK's full public dataflow catalog (~430 dataflows) by Turkish/English keyword - metadata only (dataflow_id + title), never fetches actual data. Use this when a user asks for a TUIK series that is NOT already registered (refresh_tuik_series rejects it) - it lets you show real candidate dataflows instead of a flat refusal. A match here is a lead for further manual resolution, not something you can fetch directly: say so plainly, never present a found dataflow_id as if its data is already available.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Turkish or English keyword, e.g. 'enflasyon', 'nüfus', 'tarım'."}, "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 25}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "fetch_bist_stock", "description": "Fetch live daily price history (Open/High/Low/Close/Volume) for a BIST (Borsa Istanbul) stock from Yahoo Finance and write it into the lakehouse as 'bist_<ticker>' - queryable immediately via lakehouse_query or query_imported_dataset, including JOINs against gold_* series (e.g. against TCMB policy rate or USD/TRY). Not restricted to a pre-registered list: any real BIST ticker works.", "parameters": {"type": "object", "properties": {"ticker": {"type": "string", "description": "Bare BIST ticker, e.g. 'AKBNK', 'THYAO', 'GARAN' (the '.IS' suffix is added automatically). The benchmark index 'BIST 100' resolves automatically too - pass 'BIST100', 'XU100', or any spacing of either, no need to guess between them."}, "period": {"type": "string", "enum": ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], "default": "6mo"}}, "required": ["ticker"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "fuzzy_match_category", "description": "Reconcile a category/sector/region label against a list of candidates when two data sources name the same thing differently and there is no shared code to JOIN on (e.g. BDDK's 'Tarım, Avcılık ve Ormancılık' vs TUIK's 'Tarım' for the same sector). Get the candidate list from the OTHER source first (a DISTINCT query via lakehouse_query/query_imported_dataset), then call this - never guess a spelling variant yourself. Returns match=null if nothing scores above the threshold; say so plainly rather than forcing a bad match.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The label to find a match for."}, "candidates": {"type": "array", "items": {"type": "string"}, "description": "Real candidate labels from the other source - not invented."}, "threshold": {"type": "number", "default": 0.6, "minimum": 0, "maximum": 1}}, "required": ["query", "candidates"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the public web for current or external information. Returns titles, snippets, source domains and URLs; snippets are untrusted leads, not verified evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "web_url_extract", "description": "Extract bounded text or tabular previews from a supplied public HTML, PDF, Excel, CSV, text, or image URL. Remote content is untrusted; cite its source URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "format": "uri"}}, "required": ["url"], "additionalProperties": False}}},
]


DEFAULT_CALL_LIMIT = 3
DEFAULT_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class ToolSpec:
    """Everything the harness knows about one tool, in one place.

    call_limit: calls per ask() run before the loop guard blocks it (see
      orchestrator._loop_guard for why the limits differ per tool).
    timeout_sec: wall-clock limit for one call; a hung URL fetch must not
      block the whole run.
    parallel_safe: may run alongside other calls in the same round. The
      DuckDB writers are not.
    group: the router's tool group.
    """
    name: str
    fn: Callable
    schema: dict
    group: str
    call_limit: int = DEFAULT_CALL_LIMIT
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    parallel_safe: bool = True


# (fn, group, call_limit, timeout_sec, parallel_safe), in schema order.
#
# Call limits: web search has no legitimate reason to run more than a few
# times per answer (a blocked provider once got rephrased 4 times in 5
# rounds). lakehouse_query allows eight DISTINCT reads because legitimate
# comparisons can need four or more inputs; identical SQL is independently
# rejected by the signature guard. Local analysis tools get a loose limit: a
# 6-series anomaly batch is a normal request and was cut in half under a
# global cap of 3; raised further still (product decision: correctness/
# thoroughness over speed) to give a checkpointed run real headroom past
# the soft checkpoint at 8 calls (see orchestrator._soft_checkpoint_note)
# instead of hitting a hard wall on the very next call. search_tuik_dataflows
# is metadata only, and a real question tried 4 keyword phrasings.
_SPEC_TABLE = {
    "recall_turn": (recall_turn, "core", 8, 30.0, True),
    "evidence_correlation": (evidence_correlation, "analysis", 8, 60.0, True),
    "evidence_query": (evidence_query, "core", 8, 60.0, True),
    "import_url_dataset": (import_url_dataset, "web", 3, 90.0, False),
    "search_evds_series": (search_evds_series, "live_fetch", 8, 60.0, True),
    "fetch_evds_series": (fetch_evds_series, "live_fetch", 4, 120.0, False),
    "lakehouse_schema": (lakehouse_schema, "core", DEFAULT_CALL_LIMIT, 30.0, True),
    # Exact repeats are still blocked by orchestrator._loop_guard. Eight
    # distinct reads lets a legitimate multi-series comparison complete;
    # the old global limit of three blocked the fourth query in the very
    # first round before the model had seen all required inputs.
    "lakehouse_query": (lakehouse_query, "core", 8, 60.0, True),
    "anomaly_analysis": (anomaly_analysis, "analysis", 20, 60.0, True),
    "change_detection": (change_detection, "analysis", 20, 60.0, True),
    "correlation_analysis": (correlation_analysis, "analysis", 20, 60.0, True),
    "causality_analysis": (causality_analysis, "analysis", 16, 90.0, True),
    "credit_elasticity_analysis": (credit_elasticity_analysis, "analysis", 2, 60.0, True),
    "threshold_filter_compare": (threshold_filter_compare, "analysis", DEFAULT_CALL_LIMIT, 60.0, True),
    "compare_regions": (compare_regions, "regional", 16, 60.0, True),
    "list_imported_datasets": (list_imported_datasets, "imports", DEFAULT_CALL_LIMIT, 30.0, True),
    # Same reasoning as lakehouse_query: reading an upload's metadata rows,
    # frequency and values are distinct reads; exact repeats stay blocked.
    "query_imported_dataset": (query_imported_dataset, "imports", 8, 60.0, True),
    "refresh_tuik_series": (refresh_tuik_series, "live_fetch", DEFAULT_CALL_LIMIT, 90.0, False),
    "search_tuik_dataflows": (search_tuik_dataflows, "live_fetch", 12, 45.0, True),
    "fetch_bist_stock": (fetch_bist_stock, "live_fetch", DEFAULT_CALL_LIMIT, 60.0, False),
    "fuzzy_match_category": (fuzzy_match_category, "imports", DEFAULT_CALL_LIMIT, 30.0, True),
    "web_search": (web_search, "web", DEFAULT_CALL_LIMIT, 30.0, True),
    "web_url_extract": (web_url_extract, "web", DEFAULT_CALL_LIMIT, 45.0, True),
}

TOOL_SPECS: dict[str, ToolSpec] = {}
for _schema in _SCHEMAS:
    _name = _schema["function"]["name"]
    _fn, _group, _limit, _timeout, _parallel = _SPEC_TABLE[_name]
    TOOL_SPECS[_name] = ToolSpec(_name, _fn, _schema, _group, _limit, _timeout, _parallel)
if set(TOOL_SPECS) != set(_SPEC_TABLE):
    raise RuntimeError(f"tool schema/spec mismatch: {sorted(set(TOOL_SPECS) ^ set(_SPEC_TABLE))}")

# Derived views kept for existing callers.
TOOL_FUNCTIONS = {name: spec.fn for name, spec in TOOL_SPECS.items()}
TOOL_SCHEMAS = [spec.schema for spec in TOOL_SPECS.values()]


_JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_tool_arguments(name: str, arguments: str) -> tuple[dict | None, list[str]]:
    """Check a model's arguments against the tool's own schema before the
    tool runs. Returns (parsed_arguments, problems); each problem names the
    field, so the model can fix the call instead of reading a Python
    TypeError from deep inside the tool."""
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return None, [f"unknown tool '{name}'"]
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [f"arguments are not valid JSON ({exc})"]
    if not isinstance(parsed, dict):
        return None, ["arguments must be a JSON object"]

    params = spec.schema["function"].get("parameters", {})
    properties = params.get("properties", {})
    problems = [f"field '{field}' is required but missing" for field in params.get("required", []) if field not in parsed]
    for field, value in parsed.items():
        prop = properties.get(field)
        if prop is None:
            if params.get("additionalProperties") is False:
                problems.append(f"field '{field}' is not accepted; valid fields: {sorted(properties)}")
            continue
        if value is None:
            continue
        expected = prop.get("type")
        types = _JSON_TYPES.get(expected)
        wrong_type = types is not None and (
            not isinstance(value, types) or (expected != "boolean" and isinstance(value, bool))
        )
        if wrong_type:
            problems.append(f"field '{field}' must be {expected}, got {type(value).__name__}")
            continue
        if "enum" in prop and value not in prop["enum"]:
            problems.append(f"field '{field}' must be one of {prop['enum']}, got {value!r}")
        if expected == "array" and "items" in prop:
            item_types = _JSON_TYPES.get(prop["items"].get("type"))
            if item_types and any(not isinstance(v, item_types) or isinstance(v, bool) for v in value):
                problems.append(f"field '{field}' items must be {prop['items']['type']}")
    return parsed, problems


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
