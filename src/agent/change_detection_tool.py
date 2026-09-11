"""C6 - Change Detection Tool.

Finds trend/regime break points in one ordered time series and
describes the segments between them. Network-free.

Two detectors, per the roadmap:
  - pelt           : ruptures' PELT with the rbf kernel cost
                     (rpt.Pelt(model="rbf").fit(signal).predict(pen=...)).
                     rbf reacts to a change in the signal's distribution,
                     not just its mean, so it copes with trending series
                     without over-segmenting them.
  - moving_average : the simple method - mean of the `min_segment`
                     points after each candidate vs the points before
                     it, in robust noise-sigma units, above a threshold;
                     local maxima only. Also the fallback if ruptures is
                     not installed.

Penalty calibration (pen = k * log(n), k from `sensitivity`) was done on
all 24 gold_fact_long series: the default k=0.7 gives 1-4 breaks per
series, landing on real, known events - policy rate 2023-09 (inside
the June-Dec 2023 hiking staircase), USD/TRY 2022-01 / 2023-06 /
2025-03, house price index (TP.KFE.TR) 2022-06 / 2023-08 / 2025-02.
jump=1 so dates are exact (ruptures' default jump=5 only allows a
break every 5th period). An l2 cost with a noise-sigma BIC penalty
was tried first and rejected: on smooth series (e.g. the NPL ratio) the
noise sigma from first differences is tiny, so it cut 10 breaks at every
penalty tried.

Also ported from the telecom aggregation engine (raw_n8n_code/
raw_change_detector.js): the per-series min/max/avg/last + first-half vs
second-half trend_pct/trend_direction summary, and (from the anomaly
engine) deterministic [CHANGE POINT] signature lines for the LLM to
narrate instead of recomputing.

`largest_changes` (top period-over-period moves) is kept with its
original shape - api/gold_queries.py and the web ChangeTable read it.
Input order is treated as chronological, never re-sorted.
"""

import math

import numpy as np

from src.agent.anomaly_tool import _is_missing, _num
from src.common.logging_setup import get_logger

try:
    import ruptures as rpt
except ImportError:  # optional: the moving_average detector needs nothing extra
    rpt = None

logger = get_logger("change_detection_tool")

METHODS = ("auto", "pelt", "moving_average")
SIGNALS = ("level", "pct_change", "diff")
PELT_PENALTY_FACTOR = {"low": 1.2, "medium": 0.7, "high": 0.35}  # pen = factor * log(n)
MA_THRESHOLD_SIGMA = {"low": 4.0, "medium": 3.0, "high": 2.0}
MAX_CHANGE_POINTS_RETURNED = 12


def _largest_changes(periods: list, values: list) -> list[dict]:
    """Top-10 period-over-period moves, unchanged from the original
    change_detection: a None/NaN pair is skipped, a move from exactly 0
    is reported with percent_change=None (undefined, but usually the most
    significant move, so ranked first), 0 -> 0 is not a change."""
    changes = []
    for previous, current, period in zip(values, values[1:], periods[1:]):
        if _is_missing(previous) or _is_missing(current):
            logger.debug(f"_largest_changes: skipping period={period} (missing value)")
            continue
        absolute_change = current - previous
        if previous == 0:
            if absolute_change == 0:
                continue
            changes.append({"period": period, "absolute_change": absolute_change, "percent_change": None})
            continue
        changes.append({"period": period, "absolute_change": absolute_change, "percent_change": absolute_change / abs(previous) * 100})
    changes.sort(key=lambda item: abs(item["percent_change"]) if item["percent_change"] is not None else float("inf"), reverse=True)
    return changes[:10]


def _noise_sigma(signal: np.ndarray) -> float:
    """Robust point-to-point noise estimate from first differences
    (MAD/sqrt(2)) - unaffected by a handful of level shifts. Falls back
    to mean absolute deviation when MAD is 0 (a mostly-flat series)."""
    d = np.diff(signal)
    if len(d) == 0:
        return 0.0
    dev = np.abs(d - np.median(d))
    mad = float(np.median(dev)) * 1.4826 / math.sqrt(2)
    return mad if mad > 0 else float(np.mean(dev)) / math.sqrt(2)


def _detect_pelt(signal: np.ndarray, min_segment: int, sensitivity: str, notes: list[str]) -> list[int]:
    pen = PELT_PENALTY_FACTOR[sensitivity] * math.log(len(signal))
    try:
        bkps = rpt.Pelt(model="rbf", min_size=min_segment, jump=1).fit(signal.reshape(-1, 1)).predict(pen=pen)
    except rpt.exceptions.BadSegmentationParameters:
        notes.append("series too short for PELT with this min_segment; no break points searched.")
        return []
    logger.debug(f"_detect_pelt: pen={pen:.3f}, bkps={bkps}")
    return [int(b) for b in bkps[:-1]]  # last entry is always len(signal)


def _detect_moving_average(signal: np.ndarray, min_segment: int, sensitivity: str, sigma: float) -> list[int]:
    w = min_segment
    n = len(signal)
    if sigma <= 0:
        return []
    stats = []
    for i in range(w, n - w + 1):
        stat = abs(signal[i:i + w].mean() - signal[i - w:i].mean()) / sigma
        if stat >= MA_THRESHOLD_SIGMA[sensitivity]:
            stats.append((stat, i))
    accepted: list[int] = []
    for stat, i in sorted(stats, reverse=True):  # non-maximum suppression
        if all(abs(i - j) >= w for j in accepted):
            accepted.append(i)
    logger.debug(f"_detect_moving_average: candidates={len(stats)}, accepted={sorted(accepted)}")
    return sorted(accepted)


def _signal_label(signal: str) -> str:
    return {"level": "level", "pct_change": "average per-period % change", "diff": "average per-period change"}[signal]


def _signature(cp: dict, signal: str) -> str:
    before, after = cp["before_mean"], cp["after_mean"]
    if signal == "level":
        pct = f" ({cp['shift_pct']:+.2f}%)" if cp["shift_pct"] is not None else ""
        body = f"level shifted {cp['direction']} from {before} to {after}: {cp['shift']:+}{pct}"
    elif signal == "pct_change":
        body = f"average per-period change shifted from {before:+.2f}% to {after:+.2f}% ({cp['shift']:+.2f} pp)"
    else:
        body = f"average per-period change shifted from {before:+} to {after:+} ({cp['shift']:+})"
    return (f"[CHANGE POINT] {cp['period']}: {body}; before = {cp['before_span']}, "
            f"after = {cp['after_span']}; strength {cp['strength_sigma']} sigma.")


def _summary(arr: np.ndarray) -> dict:
    """JS computeStats port: first-half vs second-half average, >5% =
    increasing, <-5% = decreasing."""
    mid = len(arr) // 2 or 1
    first_avg, second_avg = float(arr[:mid].mean()), float(arr[mid:].mean()) if len(arr) > mid else float(arr[:mid].mean())
    trend_pct = (second_avg - first_avg) / abs(first_avg) * 100 if first_avg != 0 else 0.0
    return {
        "min": _num(arr.min()), "max": _num(arr.max()), "avg": _num(arr.mean()), "last": _num(arr[-1]),
        "first_half_avg": _num(first_avg), "second_half_avg": _num(second_avg),
        "trend_pct": _num(trend_pct, 2),
        "trend_direction": "increasing" if trend_pct > 5 else "decreasing" if trend_pct < -5 else "stable",
    }


def change_detection(periods: list[str], values: list[float], method: str = "auto", signal: str = "level",
                     sensitivity: str = "medium", min_segment: int = 3) -> dict:
    """Detect break points where the series' behaviour changes regime.

    signal: 'level' (default - breaks in the level itself), 'pct_change'
    (breaks in the growth pace - e.g. 'when did inflation accelerate'),
    'diff' (same, in absolute units; use for series that can be <= 0).
    sensitivity: 'low' | 'medium' | 'high' - fewer/more break points.
    min_segment: shortest allowed regime, in periods."""
    logger.info(f"change_detection start: n={len(values) if isinstance(values, list) else '?'}, method={method}, "
                f"signal={signal}, sensitivity={sensitivity}, min_segment={min_segment}")

    if not isinstance(periods, list) or not isinstance(values, list) or len(periods) != len(values) or len(values) < 2:
        logger.error("change_detection: bad input lengths")
        raise ValueError("periods and values must have equal length and at least 2 points")
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {list(SIGNALS)}, got {signal!r}")
    if sensitivity not in PELT_PENALTY_FACTOR:
        raise ValueError(f"sensitivity must be one of {list(PELT_PENALTY_FACTOR)}, got {sensitivity!r}")
    if not isinstance(min_segment, int) or isinstance(min_segment, bool) or min_segment < 2:
        raise ValueError(f"min_segment must be an integer >= 2, got {min_segment!r}")

    notes: list[str] = []
    largest = _largest_changes(periods, values)

    clean = []
    for p, v in zip(periods, values):
        if _is_missing(v):
            continue
        try:
            clean.append((str(p), float(v)))
        except (TypeError, ValueError):
            raise ValueError(f"value for period {p!r} is not numeric: {v!r}") from None
    if len(clean) < len(values):
        notes.append(f"{len(values) - len(clean)} missing value(s) skipped.")
    clean_periods = [p for p, _ in clean]
    arr = np.array([v for _, v in clean], dtype="float64")

    if signal == "pct_change" and np.any(arr[:-1] == 0):
        notes.append("pct_change undefined where a value is 0; used diff instead.")
        signal = "diff"
    if signal == "level":
        sig, sig_periods = arr, clean_periods
    elif signal == "pct_change":
        sig, sig_periods = np.diff(arr) / np.abs(arr[:-1]) * 100, clean_periods[1:]
    else:
        sig, sig_periods = np.diff(arr), clean_periods[1:]

    used = method
    if used in ("auto", "pelt") and rpt is None:
        notes.append("ruptures is not installed; used the moving_average detector instead.")
        used = "moving_average"
    elif used == "auto":
        used = "pelt"

    sigma = _noise_sigma(sig) if len(sig) else 0.0
    if len(sig) < 2 * min_segment + 1:
        notes.append(f"need >= {2 * min_segment + 1} usable points to search for break points (min_segment={min_segment}), got {len(sig)}.")
        bkps = []
    elif np.ptp(sig) == 0:
        notes.append("signal is constant; no break points possible.")
        bkps = []
    elif used == "pelt":
        bkps = _detect_pelt(sig, min_segment, sensitivity, notes)
    else:
        bkps = _detect_moving_average(sig, min_segment, sensitivity, sigma)

    bounds = [0, *bkps, len(sig)]
    segments = []
    offset = len(clean_periods) - len(sig)  # 1 for pct_change/diff: sig[i] belongs to clean period i+1
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        segments.append({
            "start_period": sig_periods[start],
            "end_period": sig_periods[end - 1],
            "n_periods": end - start,
            "mean": _num(sig[start:end].mean()),
            "level_start": _num(arr[start + offset]),
            "level_end": _num(arr[end - 1 + offset]),
        })

    change_points = []
    for prev, seg in zip(segments, segments[1:]):
        shift = seg["mean"] - prev["mean"]
        change_points.append({
            "period": seg["start_period"],
            "direction": "up" if shift > 0 else "down",
            "before_mean": prev["mean"],
            "after_mean": seg["mean"],
            "shift": _num(shift),
            "shift_pct": _num(shift / abs(prev["mean"]) * 100, 2) if signal == "level" and prev["mean"] else None,
            "strength_sigma": _num(abs(shift) / sigma, 2) if sigma > 0 else None,
            "before_span": f"{prev['start_period']}..{prev['end_period']} ({prev['n_periods']} periods)",
            "after_span": f"{seg['start_period']}..{seg['end_period']} ({seg['n_periods']} periods)",
        })
    if len(change_points) > MAX_CHANGE_POINTS_RETURNED:
        notes.append(f"{len(change_points)} change points found; only the {MAX_CHANGE_POINTS_RETURNED} strongest are listed.")
        keep = sorted(change_points, key=lambda c: c["strength_sigma"] or 0, reverse=True)[:MAX_CHANGE_POINTS_RETURNED]
        change_points = [c for c in change_points if c in keep]

    latest = segments[-1] if segments else None
    latest_regime = None
    if latest:
        latest_regime = {"since": latest["start_period"], "n_periods": latest["n_periods"], "mean": latest["mean"],
                         "level_start": latest["level_start"], "level_end": latest["level_end"],
                         "previous_mean": segments[-2]["mean"] if len(segments) > 1 else None}

    detail = {"pelt": f"ruptures PELT, rbf cost, pen={PELT_PENALTY_FACTOR[sensitivity]}*log(n)",
              "moving_average": f"moving-average before/after comparison, window={min_segment}, threshold={MA_THRESHOLD_SIGMA[sensitivity]} noise-sigma"}[used]
    result = {
        "method": used,
        "method_detail": f"{detail}; signal={_signal_label(signal)}",
        "signal": signal,
        "sensitivity": sensitivity,
        "min_segment": min_segment,
        "n_points": len(values),
        "change_points": change_points,
        "segments": segments,
        "latest_regime": latest_regime,
        "summary": _summary(arr) if len(arr) else None,
        "largest_changes": largest,
        "event_signatures": [_signature(cp, signal) for cp in change_points],
        "notes": notes,
        "warning": "Change points are statistical breaks in the series' behaviour, dated to the FIRST period of the new regime. They show WHEN it changed, not WHY - pair with causality/web tools for an explanation. Interpret event_signatures; do not recompute the numbers.",
    }
    logger.info(f"change_detection done: method={used}, signal={signal}, change_points={[c['period'] for c in change_points]}")
    return result


CHANGE_DETECTION_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "change_detection",
    "description": (
        "C6 Change Detection Tool: find trend/regime break points in ONE ordered time series (ruptures PELT) and describe the regimes "
        "between them (mean, start/end level), the latest regime ('since when'), a first-half vs second-half trend summary, and the "
        "largest single period-over-period moves. Use for 'trend kirilmasi / ne zaman degisti / rejim / kirilma noktasi' questions. "
        "signal='pct_change' finds breaks in growth pace (e.g. when inflation or credit growth accelerated). Answer from 'event_signatures'."
    ),
    "parameters": {"type": "object", "properties": {
        "periods": {"type": "array", "items": {"type": "string"}, "description": "Chronological periods, e.g. 'YYYY-MM'."},
        "values": {"type": "array", "items": {"type": "number"}},
        "method": {"type": "string", "enum": list(METHODS), "default": "auto"},
        "signal": {"type": "string", "enum": list(SIGNALS), "default": "level"},
        "sensitivity": {"type": "string", "enum": list(PELT_PENALTY_FACTOR), "default": "medium", "description": "'low' = only major breaks, 'high' = more breaks."},
        "min_segment": {"type": "integer", "default": 3, "description": "Shortest allowed regime, in periods."},
    }, "required": ["periods", "values"], "additionalProperties": False},
}}
