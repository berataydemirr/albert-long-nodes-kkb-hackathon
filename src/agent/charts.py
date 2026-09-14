"""Derive chart specs from the agent's tool results.

WHY THIS EXISTS:
The target audience has basic financial literacy, not analyst training. A
66-row table of monthly rates answers the question technically and
communicates almost nothing; the same data as a line answers it at a
glance. So every answer that produced chartable numbers gets a chart
alongside the text, without the model having to ask for one.

The specs are built from tool output that ALREADY exists in the trace -
the model is not asked to emit chart JSON. That matters for two measured
reasons: anything the model generates is wall time (this is a reasoning
model, and generated tokens are the bottleneck), and a model transcribing
numbers into a second format is a chance to transcribe them wrong. The
numbers here are the same objects the tool returned.

FORM CHOICES (deliberate, see the project's dataviz rules):
  - Province rankings -> horizontal bar. Turkish province names are long;
    horizontal keeps them readable without rotation. Provinces are nominal
    categories, so every bar is ONE color - a darker-where-bigger ramp
    would double-encode length as hue and say nothing new.
  - Time series -> line, one hue.
  - Anomalies / largest changes -> the same line with the flagged periods
    emphasised: the series in a recessive gray, the points that matter in
    the accent. Emphasis, not a second categorical hue.
  - Two series (correlation/causality) -> TWO charts, never one plot with
    two y-axes. A dual axis invents a correlation by choosing where the
    scales line up; small multiples let the reader compare shapes honestly.

Colours are decided in the frontend from the project palette; this module
only says which points deserve emphasis.
"""

from src.agent.series_resolver import resolve_series
from src.common.logging_setup import get_logger

logger = get_logger("charts")

# More than this and the answer becomes a slideshow the reader scrolls
# past. Three is enough for the widest real case (two compared series plus
# a ranking).
MAX_CHARTS_PER_ANSWER = 3

# A line needs enough points to have a shape; below this the numbers speak
# better as text, and a 2-point "trend" is a misleading form.
MIN_POINTS_FOR_LINE = 4

# Bars past this stop being comparable and start being a table.
MAX_BARS = 12

# Period-ish and value-ish column names in lakehouse_query results, in the
# casing the gold layer actually uses.
_PERIOD_COLUMNS = ("Period", "period", "Donem", "Dönem")

# The regional metric names are slugs the tool takes as input ("npl",
# "kisi_basi_kredi"). They are fine as an API surface and useless as a
# chart title for a reader who is not an analyst, so titles get the plain
# Turkish name. Anything unmapped falls back to the slug with underscores
# opened up, which is still readable.
_METRIC_LABELS = {
    "kredi": "Toplam nakdi kredi",
    "nakdi_kredi": "Nakdi kredi",
    "gayrinakdi_kredi": "Gayrinakdi kredi",
    "takipteki_alacak": "Takipteki alacak",
    "npl": "Takipteki alacak oranı",
    "mevduat": "Toplam mevduat",
    "mevduat_tasarruf": "Tasarruf mevduatı",
    "mevduat_tasarruf_tl": "Tasarruf mevduatı (TL)",
    "mevduat_tasarruf_dth": "Tasarruf mevduatı (döviz)",
    "mevduat_diger": "Diğer mevduat",
    "konut_kredisi": "Konut kredisi",
    "tasit_kredisi": "Taşıt kredisi",
    "kmh": "Kredili mevduat hesabı",
    "diger_tuketici_kredisi": "Diğer tüketici kredileri",
    "kredi_karti": "Bireysel kredi kartları",
    "takip_konut_kredisi": "Takipteki konut kredisi",
    "takip_tasit_kredisi": "Takipteki taşıt kredisi",
    "takip_diger_tuketici": "Takipteki diğer tüketici kredileri",
    "kredi_mevduat_orani": "Kredi / mevduat oranı",
    "kredi_tasarruf_orani": "Kredi / tasarruf mevduatı oranı",
    "konut_takip_orani": "Konut kredisi takip oranı",
    "tasit_takip_orani": "Taşıt kredisi takip oranı",
    "diger_tuketici_takip_orani": "Diğer tüketici kredileri takip oranı",
    "insaat_takip_orani": "İnşaat kredileri takip oranı",
    "tekstil_takip_orani": "Tekstil kredileri takip oranı",
    "turizm_takip_orani": "Turizm kredileri takip oranı",
    "ziraat_takip_orani": "Ziraat kredileri takip oranı",
    "enerji_takip_orani": "Enerji kredileri takip oranı",
    "gida_takip_orani": "Gıda kredileri takip oranı",
    "sube_sayisi": "Yurt içi şube sayısı",
    "subeye_dusen_nufus": "Şube başına düşen nüfus",
    "kisi_basi_kredi": "Kişi başı nakdi kredi",
    "kisi_basi_takipteki_alacak": "Kişi başı takipteki alacak",
    "kisi_basi_tasarruf_mevduati": "Kişi başı tasarruf mevduatı",
    "kisi_basi_mevduat": "Kişi başı toplam mevduat",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, str(metric).replace("_", " "))


def _series_chart(seri_kod: str, highlight_periods: set[str] | None = None,
                  title_suffix: str = "") -> dict | None:
    """Read a series straight from the gold layer and shape it as a line.

    Resolving here rather than asking the model for the numbers costs one
    ~15 ms database read and removes a transcription step entirely.
    """
    try:
        resolved = resolve_series(seri_kod)
    except Exception as exc:
        # A chart is an enhancement - never let one failing take the
        # answer down with it.
        logger.warning(f"_series_chart: could not resolve {seri_kod!r} for charting: {exc}")
        return None

    periods, values = resolved["periods"], resolved["values"]
    if len(periods) < MIN_POINTS_FOR_LINE:
        logger.info(f"_series_chart: {seri_kod!r} has {len(periods)} point(s), too few to plot")
        return None

    highlight = highlight_periods or set()
    points = [
        {"label": period, "value": value, "highlight": period in highlight}
        for period, value in zip(periods, values)
    ]
    highlighted = sum(1 for p in points if p["highlight"])

    title = resolved["seri_adi"] or seri_kod
    if title_suffix:
        title = f"{title} — {title_suffix}"

    logger.info(
        f"_series_chart: {seri_kod!r} -> {len(points)} point(s), "
        f"{highlighted} highlighted, unit={resolved['birim']!r}"
    )
    return {
        "kind": "line",
        "title": title,
        "unit": resolved["birim"],
        "subtitle": f"{periods[0]} – {periods[-1]}",
        "source_note": f"gold_fact_long · {seri_kod}",
        "points": points,
    }


def _chart_from_compare_regions(payload: dict) -> dict | None:
    ranked = payload.get("ranked") or []
    if not ranked:
        return None

    bars = []
    for row in ranked[:MAX_BARS]:
        note = None
        if row.get("growth_pct") is not None:
            note = f"{row['growth_pct']:+.1f}%"
        elif row.get("share_pct") is not None:
            note = f"%{row['share_pct']:.1f}"
        bars.append({"label": row.get("bolge", "?"), "value": row.get("value", 0), "note": note})

    metric = payload.get("metric", "")
    period = payload.get("period", "")
    logger.info(f"_chart_from_compare_regions: metric={metric!r} period={period!r} -> {len(bars)} bar(s)")
    return {
        "kind": "bar",
        "title": f"{_metric_label(metric)} — il sıralaması",
        "unit": payload.get("birim"),
        "subtitle": f"{period} · en yüksek {len(bars)} il"
                    + (f" · {payload['compare_to_period']} dönemine göre değişim"
                       if payload.get("compare_to_period") else ""),
        "source_note": "gold_regional_long · BDDK FinTürk",
        "bars": bars,
    }


def _chart_from_lakehouse_query(payload: dict) -> dict | None:
    """Plot a query result only when it is unambiguously a time series.

    Deliberately conservative: the model can write any SQL it likes, and a
    chart drawn from a shape this code guessed at would be worse than no
    chart. It must have a period-like column, exactly one numeric column,
    and enough rows to have a shape.
    """
    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not columns or len(rows) < MIN_POINTS_FOR_LINE:
        return None

    period_col = next((c for c in columns if c in _PERIOD_COLUMNS), None)
    if period_col is None:
        logger.debug(f"_chart_from_lakehouse_query: no period column in {columns}, skipping")
        return None

    numeric_cols = [
        c for c in columns
        if c != period_col and all(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool) for r in rows)
    ]
    if len(numeric_cols) != 1:
        logger.debug(
            f"_chart_from_lakehouse_query: need exactly one numeric column, found {numeric_cols}; skipping"
        )
        return None

    value_col = numeric_cols[0]
    points = [{"label": str(r.get(period_col)), "value": r.get(value_col), "highlight": False} for r in rows]
    points.sort(key=lambda p: p["label"])

    logger.info(f"_chart_from_lakehouse_query: {period_col}/{value_col} -> {len(points)} point(s)")
    return {
        "kind": "line",
        "title": value_col.replace("_", " "),
        "unit": None,
        "subtitle": f"{points[0]['label']} – {points[-1]['label']}",
        "source_note": "lakehouse_query",
        "points": points,
    }


def _highlight_periods_from(payload: dict) -> set[str]:
    """Periods an analysis tool flagged as interesting, whatever it called
    them - anomalies and largest_changes use different key names for the
    same idea."""
    periods = {a.get("period") for a in (payload.get("anomalies") or []) if a.get("period")}
    periods |= {c.get("period") for c in (payload.get("largest_changes") or [])[:3] if c.get("period")}
    return periods


def derive_charts(trace: list[dict]) -> list[dict]:
    """Build chart specs from a finished run's tool trace.

    Never raises: a chart is an enhancement to an answer that is already
    correct on its own, so any failure here logs and yields nothing rather
    than turning a good answer into an error.
    """
    charts: list[dict] = []
    seen_titles: set[str] = set()

    def add(spec: dict | None) -> None:
        if spec is None or len(charts) >= MAX_CHARTS_PER_ANSWER:
            return
        # The model often calls the same analysis twice on one series; one
        # chart of it is informative, two is noise.
        if spec["title"] in seen_titles:
            logger.debug(f"derive_charts: skipping duplicate chart {spec['title']!r}")
            return
        seen_titles.add(spec["title"])
        charts.append(spec)

    for entry in trace:
        if entry.get("error") or entry.get("blocked"):
            continue
        name = entry.get("name")
        payload = entry.get("output_json")
        if not isinstance(payload, dict):
            continue

        try:
            if name == "compare_regions":
                add(_chart_from_compare_regions(payload))

            elif name in ("anomaly_analysis", "change_detection"):
                source = payload.get("source") or {}
                if source.get("seri_kod"):
                    suffix = "aykırı dönemler" if name == "anomaly_analysis" else "en büyük değişimler"
                    add(_series_chart(source["seri_kod"], _highlight_periods_from(payload), suffix))

            elif name in ("correlation_analysis", "causality_analysis"):
                # Two charts, never one plot with two y-axes.
                for side in ("x_source", "y_source"):
                    source = payload.get(side) or {}
                    if source.get("seri_kod"):
                        add(_series_chart(source["seri_kod"]))

            elif name == "lakehouse_query":
                add(_chart_from_lakehouse_query(payload))

        except Exception:
            logger.exception(f"derive_charts: failed while charting '{name}', continuing without it")

    logger.info(f"derive_charts: produced {len(charts)} chart(s) from {len(trace)} tool call(s)")
    return charts
