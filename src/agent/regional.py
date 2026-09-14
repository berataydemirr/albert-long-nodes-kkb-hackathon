"""Province-level credit and NPL access for the agent.

WHY THIS EXISTS:
gold_finturk_regional_quarterly holds 82 rows per quarter - 81 provinces
plus "YURT DIŞI" - with cash loans and the non-performing-loan ratio, from
2021-12 to 2026-6. None of it was reachable from the agent: every row in
gold_fact_long carries Bolge='TR', so the main fact table has no regional
dimension at all. The data was collected, cleaned and then never exposed.

Two traps this module exists to handle, both of which produce a wrong
answer silently rather than an error:

  1. Periods are "YYYY-Q" with no zero padding, so string ordering is
     wrong WITHIN a year: '2021-12' < '2021-3' < '2021-6' < '2021-9'.
     A plain MAX(Period) or ORDER BY Period returns the wrong quarter.
     Every ordering here goes through _quarter_sort_key instead.

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
REGIONAL_TABLE = "gold_finturk_regional_quarterly"

# Not a province; see the module docstring.
NON_PROVINCE_LABELS = ("YURT DIŞI",)

METRICS = {
    "kredi": {
        "column": "Toplam_Nakdi_Kredi_Bin_TL",
        "unit": "bin TL",
        "label": "Toplam nakdi kredi",
    },
    "npl": {
        "column": "NPL_Orani_Yaklasik",
        "unit": "oran (0-1)",
        "label": "Takipteki alacak oranı (yaklaşık)",
    },
}

MAX_TOP_N = 25
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


def _resolve_period(requested: str | None, periods: list[str]) -> str:
    if not periods:
        logger.error("_resolve_period: regional table has no periods")
        raise ValueError(f"{REGIONAL_TABLE} is empty.")
    if requested is None:
        latest = periods[-1]
        logger.info(f"_resolve_period: defaulting to the true latest quarter {latest!r}")
        return latest
    cleaned = str(requested).strip()
    if cleaned not in periods:
        logger.warning(f"_resolve_period: {cleaned!r} not in the table")
        raise ValueError(f"Unknown period {cleaned!r}. Available: {periods}")
    logger.debug(f"_resolve_period: using requested period {cleaned!r}")
    return cleaned


def compare_regions(period: str | None = None, metric: str = "kredi", top_n: int = 10,
                    order: str = "desc", compare_to_period: str | None = None,
                    include_non_provinces: bool = False) -> dict:
    """Rank provinces for one quarter, optionally with growth against an
    earlier quarter.

    metric: "kredi" (total cash loans) or "npl" (non-performing ratio).
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

    if metric not in METRICS:
        logger.error(f"compare_regions: unknown metric {metric!r}")
        raise ValueError(f"Unknown metric {metric!r}. Use one of: {sorted(METRICS)}")
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

    spec = METRICS[metric]
    column = spec["column"]

    periods = available_periods()
    target = _resolve_period(period, periods)
    baseline = _resolve_period(compare_to_period, periods) if compare_to_period is not None else None
    if baseline is not None and _quarter_sort_key(baseline) >= _quarter_sort_key(target):
        logger.error(f"compare_regions: baseline {baseline!r} is not before target {target!r}")
        raise ValueError(f"compare_to_period ({baseline}) must be earlier than period ({target}).")

    con = _db()
    try:
        rows = con.execute(
            f'SELECT Bolge, "{column}" FROM "{REGIONAL_TABLE}" WHERE Period = ?', [target]
        ).fetchall()
        logger.info(f"compare_regions: {len(rows)} row(s) for {target!r}")

        baseline_values: dict[str, float] = {}
        if baseline is not None:
            baseline_values = {
                bolge: value for bolge, value in con.execute(
                    f'SELECT Bolge, "{column}" FROM "{REGIONAL_TABLE}" WHERE Period = ?', [baseline]
                ).fetchall()
            }
            logger.info(f"compare_regions: {len(baseline_values)} baseline row(s) for {baseline!r}")
    finally:
        con.close()

    if not rows:
        logger.error(f"compare_regions: no rows for period {target!r}")
        raise ValueError(f"No regional rows for period {target!r}.")

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
        if metric == "kredi" and total:
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
        "metric_label": spec["label"],
        "birim": spec["unit"],
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
    if metric == "kredi":
        result["total_value"] = float(total)
    if baseline is not None:
        result["compare_to_period"] = baseline

    logger.info(
        f"compare_regions: done period={target!r}, metric={metric!r}, returned={len(results)}, "
        f"top={results[0]['bolge'] if results else None}"
    )
    return result
