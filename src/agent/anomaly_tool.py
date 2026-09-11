"""C4 - Anomaly Tool.

Statistical outlier detection on one ordered time series. Pure
pandas/numpy (+ statsmodels STL for the seasonal variant), network-free.

Ported from the telecom KPI anomaly engine (raw_n8n_code/
raw_anomaly_detector.js) and adapted to macro/financial series:
  - the JS engine's "z-score vs baseline" detector      -> robust z-score
  - its direction-aware rules (drop vs surge per KPI)   -> `direction`
  - its practical-significance guard (z AND abs diff)   -> `min_deviation_pct`
  - its warning/critical severity split                 -> `severity`
  - its active/resolved status + contiguous-block cards -> `episodes`
  - its [EVENT SIGNATURE] lines (deterministic facts
    for the LLM to narrate, never to recompute)          -> `event_signatures`

Methods (all score a RESIDUAL = value vs an expected value, then use a
robust scale so one huge outlier can't inflate the scale enough to mask
itself - the reason std-based z-scores were rejected, see
test_anomaly_analysis_finds_an_outlier_when_mad_is_zero):
  - mad            : expected = series median. Fine for short/flat series,
                     but on a trending series it flags the END of the
                     trend, not real surprises.
  - rolling_zscore : expected = trailing moving MEDIAN of the previous
                     `window` points (the point itself excluded, so it
                     can't pull its own baseline; a median, not a mean,
                     so a spike doesn't inflate the baseline and make
                     the next `window` normal points look like drops).
                     Relative (%) residual
                     for strictly positive series, so exponential growth
                     (CPI, USD/TRY) isn't scored as ever-growing outliers.
  - rolling_iqr    : same residual, Tukey fences (threshold = fence
                     multiplier: 1.5 mild, 3.0 extreme).
  - stl            : expected = STL trend + seasonal (robust STL, log
                     space for positive series). Needs 2+ full seasons.
  - auto           : rolling_zscore if long enough (>= 2*window+3
                     points), else mad. See _choose_method for why stl
                     is not the auto default.

Input order is treated as chronological (same house rule as the other
tools: periods are never re-sorted, since 'YYYY-Q' strings don't sort
lexicographically).
"""

import math
import re

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.common.logging_setup import get_logger

logger = get_logger("anomaly_tool")

METHODS = ("auto", "mad", "rolling_zscore", "rolling_iqr", "stl")
DIRECTIONS = ("both", "up", "down")
DEFAULT_WINDOW = 6
CRITICAL_MULTIPLIER = 1.5  # score >= threshold * 1.5 -> critical (JS engine: 5% vs 20% past SLA)
MAX_ANOMALIES_RETURNED = 30
MAX_EPISODES_RETURNED = 15


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(x, ndigits: int = 4):
    """JSON-safe float: NaN/inf -> None, numpy scalars -> plain float."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, ndigits)


def infer_season_length(periods: list[str]) -> int | None:
    """12 for 'YYYY-MM', 4 for FinTurk's 'YYYY-Q' (Q in 3,6,9,12, not
    zero-padded), None for annual or anything unrecognised. A single
    non-padded period marks the whole series as quarterly - monthly
    periods in the gold layer are always zero-padded."""
    if not periods:
        return None
    if all(re.fullmatch(r"\d{4}-(3|6|9|12)", p) for p in periods) and any(re.fullmatch(r"\d{4}-\d", p) for p in periods):
        return 4
    if all(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", p) for p in periods):
        return 12
    return None


def _robust_scores(residual: np.ndarray) -> tuple[np.ndarray, str]:
    """Signed robust z-score of each residual around the residuals' own
    median. NaN residuals (warm-up points) stay NaN.

    MAD==0 (>= half the residuals identical - e.g. a policy rate held
    flat for months) falls back to mean absolute deviation from the
    median, which grows only linearly with one outlier and so keeps it
    visible; only a genuinely constant residual scores all-zero."""
    valid = residual[~np.isnan(residual)]
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    if mad > 0:
        return 0.6745 * (residual - center) / mad, "MAD-scaled robust z-score"
    mean_ad = float(np.mean(np.abs(valid - center)))
    if mean_ad > 0:
        logger.warning(f"_robust_scores: MAD=0, falling back to mean-absolute-deviation scale (mean_ad={mean_ad})")
        return (residual - center) / mean_ad, "mean-absolute-deviation scale (MAD was 0)"
    logger.warning("_robust_scores: zero variance in residuals, no anomalies possible")
    return np.where(np.isnan(residual), np.nan, 0.0), "degenerate: zero variance, no anomalies possible"


def _iqr_scores(residual: np.ndarray) -> tuple[np.ndarray, str]:
    """Signed distance outside the Tukey box, in IQR units (0 inside the
    box). Falls back to robust z when IQR==0."""
    valid = residual[~np.isnan(residual)]
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        scores, scale = _robust_scores(residual)
        return scores, f"IQR was 0, fell back to {scale}"
    scores = np.where(residual > q3, (residual - q3) / iqr, np.where(residual < q1, (residual - q1) / iqr, 0.0))
    return np.where(np.isnan(residual), np.nan, scores), "Tukey IQR fences (score = IQRs outside the box)"


def _expected_mad(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    median = float(np.median(values))
    expected = np.full(len(values), median)
    return expected, values - expected, "expected = series median"


def _expected_rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Trailing moving median of the previous `window` points (shift(1):
    the point being scored never sits inside its own baseline). Residual
    is relative for strictly positive series, absolute otherwise; the
    residuals' median is folded into `expected` so a steady uptrend's
    typical 'above the trailing baseline' gap counts as normal."""
    series = pd.Series(values)
    baseline = series.rolling(window=window, min_periods=window).median().shift(1).to_numpy()
    relative = bool(np.all(values > 0))
    if relative:
        residual = values / baseline - 1.0
        center = float(np.nanmedian(residual))
        expected = baseline * (1.0 + center)
        detail = f"expected = {window}-period trailing moving median (relative % deviation)"
    else:
        residual = values - baseline
        center = float(np.nanmedian(residual))
        expected = baseline + center
        detail = f"expected = {window}-period trailing moving median (absolute deviation)"
    return expected, residual, detail


def _expected_stl(values: np.ndarray, season_length: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Robust STL with a PERIODIC seasonal (R's s.window="periodic": a
    huge seasonal window, degree 0 - one fixed seasonal shape). With only
    4-6 years of data each month's cycle-subseries has 4-6 points, and
    STL's default seasonal LOESS (window 7) fits them almost exactly,
    absorbing the noise into 'seasonality' - residual MAD collapses and
    ordinary points score 5-15 (confirmed: a synthetic sine + one spike
    flagged 15 points with the default, exactly the 1 spike with this)."""
    log_space = bool(np.all(values > 0))
    fit_input = np.log(values) if log_space else values
    fit = STL(fit_input, period=season_length, seasonal=10 * len(values) + 1, seasonal_deg=0, robust=True).fit()
    fitted = fit.trend + fit.seasonal
    expected = np.exp(fitted) if log_space else fitted
    residual = fit_input - fitted
    detail = f"expected = robust STL trend + periodic seasonal (period={season_length}{', log space' if log_space else ''})"
    return expected, residual, detail


def _choose_method(method: str, n: int, window: int, season_length: int | None, notes: list[str]) -> str:
    rolling_min = 2 * window + 3
    stl_min = 2 * season_length + 1 if season_length else None

    if method == "auto":
        # Deliberately NOT stl: calibrated on all 24 gold series, STL's
        # flexible trend leaves tiny residuals on smooth trending macro
        # series, so its robust scale shrinks - even with the periodic
        # seasonal it flagged 10-19 points each on CPI/house prices/
        # policy rate (whole regime shifts, which are change_detection's
        # job). rolling_zscore flagged 0-11 with economically meaningful
        # deviations (USD/TRY 2021-11, 2023-06). stl stays available on
        # request for genuinely seasonal flows.
        if n >= rolling_min:
            return "rolling_zscore"
        return "mad"
    if method == "stl" and (season_length is None or n < stl_min):
        reason = "no recognised seasonal period (needs 'YYYY-MM' or 'YYYY-Q' periods)" if season_length is None else f"needs >= {stl_min} points, got {n}"
        notes.append(f"stl not possible ({reason}); used rolling_zscore/mad instead.")
        return "rolling_zscore" if n >= rolling_min else "mad"
    if method in ("rolling_zscore", "rolling_iqr") and n < rolling_min:
        notes.append(f"{method} needs >= {rolling_min} points for window={window}, got {n}; used mad instead.")
        return "mad"
    return method


def _build_episodes(flagged: list[dict], clean_periods: list[str]) -> list[dict]:
    """Group anomalies on adjacent observed points into episodes (the JS
    engine's contiguous-block cards), each with status and recovery."""
    position = {p: i for i, p in enumerate(clean_periods)}
    last_pos = len(clean_periods) - 1
    blocks: list[list[dict]] = []
    for a in flagged:
        if blocks and position[a["period"]] - position[blocks[-1][-1]["period"]] <= 1:
            blocks[-1].append(a)
        else:
            blocks.append([a])

    episodes = []
    for block in blocks:
        peak = max(block, key=lambda a: abs(a["score"]))
        directions = {a["direction"] for a in block}
        end_pos = position[block[-1]["period"]]
        episodes.append({
            "start_period": block[0]["period"],
            "end_period": block[-1]["period"],
            "n_points": len(block),
            "direction": directions.pop() if len(directions) == 1 else "mixed",
            "severity": "critical" if any(a["severity"] == "critical" for a in block) else "warning",
            "peak_period": peak["period"],
            "peak_value": peak["value"],
            "peak_expected": peak["expected"],
            "peak_deviation_pct": peak["deviation_pct"],
            "peak_score": peak["score"],
            "status": "active" if end_pos == last_pos else "resolved",
            "recovered_on": clean_periods[end_pos + 1] if end_pos < last_pos else None,
        })
    return episodes


def _event_signature(ep: dict) -> str:
    word = {"up": "above", "down": "below", "mixed": "above and below"}[ep["direction"]]
    span = ep["start_period"] if ep["start_period"] == ep["end_period"] else f"{ep['start_period']} -> {ep['end_period']}"
    pct = f", {ep['peak_deviation_pct']:+.2f}%" if ep["peak_deviation_pct"] is not None else ""
    text = (f"[ANOMALY EPISODE] {span}: {ep['n_points']} point(s) {word} expected level; "
            f"peak {ep['peak_period']} (value {ep['peak_value']}, expected {ep['peak_expected']}{pct}, "
            f"score {ep['peak_score']}, {ep['severity']});")
    if ep["recovered_on"]:
        text += f" back to normal from {ep['recovered_on']}."
    else:
        text += " still ongoing at the latest observation."
    return text


def anomaly_analysis(periods: list[str], values: list[float], threshold: float = 3.0, method: str = "auto",
                     direction: str = "both", window: int = DEFAULT_WINDOW, min_deviation_pct: float = 0.0) -> dict:
    """Detect statistical outliers in one ordered series; see module
    docstring for the methods. `threshold` is the robust z-score cutoff
    (for rolling_iqr: the Tukey fence multiplier). `direction` keeps only
    spikes ('up') or drops ('down'). `min_deviation_pct` additionally
    requires |value - expected| to be at least that % of expected, so a
    statistically unusual but economically trivial wiggle is ignored."""
    logger.info(f"anomaly_analysis start: n={len(values) if isinstance(values, list) else '?'}, threshold={threshold}, "
                f"method={method}, direction={direction}, window={window}, min_deviation_pct={min_deviation_pct}")

    if not isinstance(periods, list) or not isinstance(values, list) or len(periods) != len(values):
        logger.error("anomaly_analysis: periods/values must be lists of equal length")
        raise ValueError("periods and values must be lists of equal length")
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {list(DIRECTIONS)}, got {direction!r}")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold!r}")
    if not isinstance(window, int) or isinstance(window, bool) or window < 3:
        raise ValueError(f"window must be an integer >= 3, got {window!r}")
    if not isinstance(min_deviation_pct, (int, float)) or min_deviation_pct < 0:
        raise ValueError(f"min_deviation_pct must be a number >= 0, got {min_deviation_pct!r}")

    clean = []
    for p, v in zip(periods, values):
        if _is_missing(v):
            continue
        try:
            clean.append((str(p), float(v)))
        except (TypeError, ValueError):
            raise ValueError(f"value for period {p!r} is not numeric: {v!r}") from None
    missing_count = len(values) - len(clean)
    if len(clean) < 3:
        logger.error(f"anomaly_analysis: only {len(clean)} non-missing points")
        raise ValueError("periods and values must have equal length and at least 3 non-missing points")

    clean_periods = [p for p, _ in clean]
    arr = np.array([v for _, v in clean], dtype="float64")
    notes: list[str] = []
    if missing_count:
        notes.append(f"{missing_count} missing value(s) skipped.")

    season_length = infer_season_length(clean_periods)
    used = _choose_method(method, len(arr), window, season_length, notes)
    logger.debug(f"anomaly_analysis: season_length={season_length}, method_used={used}")

    if used == "mad":
        expected, residual, detail = _expected_mad(arr)
    elif used == "stl":
        expected, residual, detail = _expected_stl(arr, season_length)
    else:
        expected, residual, detail = _expected_rolling(arr, window)
        notes.append(f"first {window} point(s) are warm-up for the moving median and are not scored.")

    signed, scale_detail = _iqr_scores(residual) if used == "rolling_iqr" else _robust_scores(residual)

    flagged = []
    for i, (period, value) in enumerate(clean):
        s = signed[i]
        if np.isnan(s) or abs(s) < threshold:
            continue
        move = "up" if s > 0 else "down"
        if direction != "both" and move != direction:
            continue
        exp_i = float(expected[i])
        deviation_pct = (value - exp_i) / abs(exp_i) * 100 if exp_i != 0 else None
        if min_deviation_pct and deviation_pct is not None and abs(deviation_pct) < min_deviation_pct:
            continue
        flagged.append({
            "period": period,
            "value": _num(value),
            "expected": _num(exp_i),
            "deviation": _num(value - exp_i),
            "deviation_pct": _num(deviation_pct, 2),
            "score": round(abs(float(s)), 3),
            "direction": move,
            "severity": "critical" if abs(s) >= threshold * CRITICAL_MULTIPLIER else "warning",
        })

    episodes = _build_episodes(flagged, clean_periods)

    last_i = len(arr) - 1
    last_flag = flagged[-1] if flagged and flagged[-1]["period"] == clean_periods[-1] else None
    latest = {
        "period": clean_periods[-1],
        "value": _num(arr[last_i]),
        "expected": _num(expected[last_i]),
        "score": _num(abs(signed[last_i]), 3),
        "status": "anomalous" if last_flag else "normal",
    }

    anomalies = flagged
    truncated = len(flagged) > MAX_ANOMALIES_RETURNED
    if truncated:
        keep = {a["period"] for a in sorted(flagged, key=lambda a: a["score"], reverse=True)[:MAX_ANOMALIES_RETURNED]}
        anomalies = [a for a in flagged if a["period"] in keep]
        notes.append(f"{len(flagged)} anomalies found; only the {MAX_ANOMALIES_RETURNED} highest-scoring are listed.")
    if len(episodes) > MAX_EPISODES_RETURNED:
        notes.append(f"{len(episodes)} episodes found; only the {MAX_EPISODES_RETURNED} most severe are listed.")
        keep_ep = sorted(episodes, key=lambda e: e["peak_score"], reverse=True)[:MAX_EPISODES_RETURNED]
        episodes = [e for e in episodes if e in keep_ep]

    result = {
        "method": used,
        "method_detail": f"{detail}; {scale_detail}",
        "threshold": threshold,
        "direction": direction,
        "window": window if used in ("rolling_zscore", "rolling_iqr") else None,
        "season_length": season_length if used == "stl" else None,
        "n_points": len(values),
        "n_scored": int(np.sum(~np.isnan(signed))),
        "anomaly_count": len(flagged),
        "anomalies": anomalies,
        "episodes": episodes,
        "latest": latest,
        "event_signatures": [_event_signature(e) for e in episodes],
        "notes": notes,
        "warning": "Statistical outliers relative to the series' own history/expected level - they show WHEN the series behaved unusually, not WHY. Interpret the event_signatures; do not recompute the numbers.",
    }
    if truncated:
        result["truncated"] = True
    logger.info(f"anomaly_analysis done: method={used}, anomalies={len(flagged)}, episodes={len(episodes)}, latest={latest['status']}")
    return result


ANOMALY_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "anomaly_analysis",
    "description": (
        "C4 Anomaly Tool: find statistically unusual points (spikes/drops) in ONE ordered time series and group them into episodes "
        "with start/end, peak, severity (warning/critical), status (active/resolved) and recovery period. Default method='auto' scores "
        "deviation from a trailing moving median (median/MAD for very short series). Use method='stl' for seasonal flow series "
        "(Tip='flow', e.g. card spending, industrial production) or when the user asks to remove seasonality ('mevsimsellikten arindir'); "
        "use method='rolling_iqr' only when the user asks for IQR/quartile fences. "
        "Use for 'olagan disi / anormal / sapma / beklenmedik' questions. Answer from 'episodes' and 'event_signatures'."
    ),
    "parameters": {"type": "object", "properties": {
        "periods": {"type": "array", "items": {"type": "string"}, "description": "Chronological periods, e.g. 'YYYY-MM'."},
        "values": {"type": "array", "items": {"type": "number"}},
        "threshold": {"type": "number", "default": 3.0, "description": "Robust z-score cutoff (2.5 = more sensitive, 3.5 = stricter)."},
        "method": {"type": "string", "enum": list(METHODS), "default": "auto", "description": "auto (default) = deviation from trailing moving median; stl = seasonal flow series / 'mevsimsellikten arindir'; rolling_iqr = user asks for IQR / 'ceyrekler arasi'; mad = whole-history median, very short series."},
        "direction": {"type": "string", "enum": list(DIRECTIONS), "default": "both", "description": "'up' = only spikes, 'down' = only drops."},
        "window": {"type": "integer", "default": DEFAULT_WINDOW, "description": "Moving-average window for rolling methods."},
        "min_deviation_pct": {"type": "number", "default": 0.0, "description": "Ignore anomalies deviating less than this % from expected."},
    }, "required": ["periods", "values"], "additionalProperties": False},
}}
