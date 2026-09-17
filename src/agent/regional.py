"""Province-level credit and NPL access for the agent.

WHY THIS EXISTS:
gold_finturk_regional_quarterly holds 82 rows per quarter - 81 provinces
plus "YURT DIŞI" - with cash loans and the non-performing-loan ratio, from
2021-12 to 2026-6. None of it was reachable from the agent: every row in
gold_fact_long carries Bolge='TR', so the main fact table has no regional
dimension at all. The data was collected, cleaned and then never exposed.

Two traps this module exists to handle, both of which produce a wrong
answer silently rather than an error:

  1. Periods USED to be written "YYYY-Q" with no zero padding, which made
     string ordering wrong WITHIN a year: '2021-12' < '2021-3' < '2021-6'
     < '2021-9', so a plain MAX(Period) or ORDER BY Period returned the
     wrong quarter. The gold layer now stores the padded 'YYYY-MM' form
     (see build_gold._load_finturk_table), so string order and calendar
     order finally agree. _quarter_sort_key is kept anyway: it is what
     makes this module correct against BOTH spellings, including a
     database built before the fix, and it costs nothing.

  2. "YURT DIŞI" is not a province. It is 1.2% of national lending, which
     is large enough to land inside a top-10 list, so a ranking that
     treats it as a province is simply incorrect. It is excluded from
     province rankings by default and reported separately, so it is
     never silently dropped either.

Output is deliberately a ranked head plus totals, never the full 82 rows:
the whole point is that the model can ask about regions without 82 rows
entering the context on every call.
"""

import re

import duckdb

from config.settings import GOLD_DIR
from src.common.logging_setup import get_logger

logger = get_logger("regional")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
REGIONAL_TABLE = "gold_regional_long"

# Not a province; see the module docstring.
NON_PROVINCE_LABELS = ("YURT DIŞI",)

MAX_TOP_N = 25

# Names worth knowing without a lookup. The full set lives in the table
# (35 metrics and counting), and an unknown name comes back with the list
# - putting all of them in the tool schema would charge the model's
# context for every metric on every single request.
COMMON_METRICS = (
    "kredi", "npl", "mevduat", "konut_kredisi", "kredi_karti",
    "kisi_basi_kredi", "sube_sayisi", "kredi_mevduat_orani",
)
_PERIOD_RE = re.compile(r"^(\d{4})-(\d{1,2})$")


def _db() -> duckdb.DuckDBPyConnection:
    logger.debug(f"_db: opening {DB_PATH} (read_only=True)")
    if not DB_PATH.exists():
        logger.error(f"_db: gold database missing at {DB_PATH}")
        raise FileNotFoundError(
            f"Gold database not found: {DB_PATH}. Run `python -m src.gold.build_gold` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


def _quarter_sort_key(period: str) -> tuple[int, int]:
    """'2021-12' -> (2021, 12). Sorting on this instead of the raw string
    is the whole defence against trap 1 in the module docstring."""
    match = _PERIOD_RE.match(str(period).strip())
    if not match:
        logger.warning(f"_quarter_sort_key: unparseable period {period!r}, sorting it last")
        return (9999, 99)
    return (int(match.group(1)), int(match.group(2)))


def available_metrics() -> list[dict]:
    """Every metric the regional table carries, with its unit.

    Read from the table rather than hard-coded: the metric set is data, so
    a gold rebuild that adds one makes it reachable with no code change.
    """
    con = _db()
    try:
        rows = con.execute(
            f'SELECT DISTINCT Metrik, Birim, Toplanabilir FROM "{REGIONAL_TABLE}" ORDER BY Metrik'
        ).fetchall()
    finally:
        con.close()
    metrics = [{"metric": m, "birim": b, "toplanabilir": bool(a)} for m, b, a in rows]
    logger.info(f"available_metrics: {len(metrics)} metric(s)")
    return metrics


def available_periods() -> list[str]:
    """Every quarter in the regional table, correctly ordered."""
    con = _db()
    try:
        periods = [row[0] for row in con.execute(f'SELECT DISTINCT Period FROM "{REGIONAL_TABLE}"').fetchall()]
    finally:
        con.close()
    ordered = sorted(periods, key=_quarter_sort_key)
    logger.info(f"available_periods: {len(ordered)} quarter(s), {ordered[0] if ordered else None}..{ordered[-1] if ordered else None}")
    return ordered


def _normalize_period(period: str) -> str:
    """'2026-6' and '2026-06' are the same quarter; return the padded form.

    The gold layer now stores the padded spelling everywhere (see
    build_gold._load_finturk_table for why), but the unpadded one is still
    reachable: it is what this tool's own schema used to document, what
    earlier chat turns carry in their history, and what a person types by
    hand. Accepting both costs nothing and stops a formatting detail from
    turning into "Unknown period" in the middle of a demo.
    """
    match = _PERIOD_RE.match(str(period).strip())
    if not match:
        return str(period).strip()
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"


def _resolve_period(requested: str | None, periods: list[str]) -> str:
    if not periods:
        logger.error("_resolve_period: regional table has no periods")
        raise ValueError(f"{REGIONAL_TABLE} is empty.")
    if requested is None:
        latest = periods[-1]
        logger.info(f"_resolve_period: defaulting to the true latest quarter {latest!r}")
        return latest
    cleaned = str(requested).strip()
    if cleaned in periods:
        logger.debug(f"_resolve_period: using requested period {cleaned!r}")
        return cleaned
    # Fall back to a spelling-insensitive match before giving up, so
    # '2026-6' still finds the stored '2026-06' (and vice versa).
    normalized = _normalize_period(cleaned)
    by_normalized = {_normalize_period(p): p for p in periods}
    if normalized in by_normalized:
        resolved = by_normalized[normalized]
        logger.info(f"_resolve_period: normalized {cleaned!r} -> {resolved!r}")
        return resolved
    logger.warning(f"_resolve_period: {cleaned!r} not in the table")
    raise ValueError(f"Unknown period {cleaned!r}. Available: {periods}")


def compare_regions(period: str | None = None, metric: str = "kredi", top_n: int = 10,
                    order: str = "desc", compare_to_period: str | None = None,
                    include_non_provinces: bool = False) -> dict:
    """Rank provinces for one quarter, optionally with growth against an
    earlier quarter.

    metric: any name in the regional table - "kredi", "npl", "mevduat",
    "konut_kredisi", "kisi_basi_kredi", "sube_sayisi" and ~30 more. An
    unknown name comes back with the full list rather than a bare error.
    period / compare_to_period: "YYYY-Q" with Q in 3/6/9/12; period
    defaults to the true latest quarter.
    order: "desc" (largest first) or "asc".
    include_non_provinces: keep "YURT DIŞI" in the ranking. Off by
    default because it is not a province; its value is reported under
    "non_province_rows" either way.
    """
    logger.info(
        f"compare_regions: start period={period!r}, metric={metric!r}, top_n={top_n}, "
        f"order={order!r}, compare_to_period={compare_to_period!r}, include_non_provinces={include_non_provinces}"
    )

    known = {row["metric"]: row for row in available_metrics()}
    if metric not in known:
        logger.error(f"compare_regions: unknown metric {metric!r}")
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(known)}"
        )
    if order not in ("desc", "asc"):
        logger.error(f"compare_regions: unknown order {order!r}")
        raise ValueError("order must be 'desc' or 'asc'.")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        logger.error(f"compare_regions: invalid top_n {top_n!r}")
        raise ValueError("top_n must be a positive integer.")
    # Capped rather than rejected: the cap is what keeps this tool from
    # putting all 82 rows into the model's context.
    effective_top_n = min(top_n, MAX_TOP_N)
    if effective_top_n != top_n:
        logger.warning(f"compare_regions: top_n {top_n} capped to {MAX_TOP_N}")

    spec = known[metric]
    # Whether summing across provinces means anything. A "share of the
    # national NPL ratio" would be a number with no interpretation, so
    # shares and totals are reported only for additive metrics.
    additive = spec["toplanabilir"]
    logger.debug(f"compare_regions: metric={metric!r} birim={spec['birim']!r} additive={additive}")

    periods = available_periods()
    target = _resolve_period(period, periods)
    baseline = _resolve_period(compare_to_period, periods) if compare_to_period is not None else None
    if baseline is not None and _quarter_sort_key(baseline) >= _quarter_sort_key(target):
        logger.error(f"compare_regions: baseline {baseline!r} is not before target {target!r}")
        raise ValueError(f"compare_to_period ({baseline}) must be earlier than period ({target}).")

    con = _db()
    try:
        rows = con.execute(
            f'SELECT Bolge, Deger FROM "{REGIONAL_TABLE}" WHERE Period = ? AND Metrik = ?',
            [target, metric],
        ).fetchall()
        logger.info(f"compare_regions: {len(rows)} row(s) for {target!r}")

        baseline_values: dict[str, float] = {}
        if baseline is not None:
            baseline_values = {
                bolge: value for bolge, value in con.execute(
                    f'SELECT Bolge, Deger FROM "{REGIONAL_TABLE}" WHERE Period = ? AND Metrik = ?',
                    [baseline, metric],
                ).fetchall()
            }
            logger.info(f"compare_regions: {len(baseline_values)} baseline row(s) for {baseline!r}")
    finally:
        con.close()

    if not rows:
        logger.error(f"compare_regions: no rows for period {target!r}")
        raise ValueError(f"No rows for metric {metric!r} in period {target!r}.")

    provinces = [(b, v) for b, v in rows if b not in NON_PROVINCE_LABELS and v is not None]
    non_provinces = [
        {"bolge": b, "value": float(v)}
        for b, v in rows if b in NON_PROVINCE_LABELS and v is not None
    ]
    logger.debug(f"compare_regions: {len(provinces)} province(s), {len(non_provinces)} non-province row(s)")

    ranked_pool = provinces + [(item["bolge"], item["value"]) for item in non_provinces] \
        if include_non_provinces else provinces

    # The share denominator follows the same inclusion rule as the ranking,
    # so the percentages in a result always add up against what it ranked.
    total = sum(value for _, value in ranked_pool)
    ranked = sorted(ranked_pool, key=lambda item: item[1], reverse=(order == "desc"))

    results = []
    for rank, (bolge, value) in enumerate(ranked[:effective_top_n], start=1):
        entry = {"rank": rank, "bolge": bolge, "value": float(value)}
        # A share of a ratio is meaningless, so it is only reported for
        # additive metrics.
        if additive and total:
            entry["share_pct"] = round(value / total * 100, 2)
        if baseline is not None:
            previous = baseline_values.get(bolge)
            if previous is None:
                entry["growth_pct"] = None
                entry["growth_note"] = f"{baseline} döneminde bu bölge yok"
            elif previous == 0:
                entry["growth_pct"] = None
                entry["growth_note"] = f"{baseline} değeri 0, yüzde değişim tanımsız"
            else:
                entry["growth_pct"] = round((value - previous) / abs(previous) * 100, 2)
        results.append(entry)

    result = {
        "period": target,
        "metric": metric,
        "birim": spec["birim"],
        "toplanabilir": additive,
        "order": order,
        "province_count": len(provinces),
        "ranked": results,
        "returned": len(results),
        "non_province_rows": non_provinces,
        "note": (
            "'YURT DIŞI' bir il değildir; sıralamaya dahil edilmedi ancak "
            "non_province_rows altında raporlandı."
            if non_provinces and not include_non_provinces
            else "Sıralama yalnızca gerçek illeri içerir."
        ),
    }
    if additive:
        result["total_value"] = float(total)
    if baseline is not None:
        result["compare_to_period"] = baseline

    logger.info(
        f"compare_regions: done period={target!r}, metric={metric!r}, returned={len(results)}, "
        f"top={results[0]['bolge'] if results else None}"
    )
    return result
