"""Faz D2: Plotly grafik uretimi.

Every chart function is defensive about the data it's handed - empty,
all-null, single-point, or mismatched-length input renders a small
placeholder figure with an explanatory annotation instead of raising
(a report generator crashing on the one series that happens to be
sparse is worse than a chart panel that honestly says "veri yok").
"""
import duckdb
import pandas as pd
import plotly.graph_objects as go

from config.settings import GOLD_DIR
from src.common.logging_setup import get_logger

logger = get_logger("charts")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def query_series_from_gold(seri_kod: str) -> pd.DataFrame:
    """Fetches one Seri_Kod's (Period, Deger) rows from gold_fact_long,
    sorted. Returns an empty DataFrame (not an error) for an unknown
    Seri_Kod - the caller's chart function turns that into a placeholder."""
    logger.info(f"query_series_from_gold: seri_kod={seri_kod}")
    if not DB_PATH.exists():
        logger.error(f"query_series_from_gold: gold database missing at {DB_PATH}")
        return pd.DataFrame(columns=["Period", "Deger"])
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT Period, Deger FROM gold_fact_long WHERE Seri_Kod = ? ORDER BY Period", [seri_kod]
        ).fetchdf()
    finally:
        con.close()
    logger.debug(f"query_series_from_gold: {len(df)} rows for {seri_kod}")
    return df


def series_line_chart(periods: list, values: list, title: str, y_label: str = "") -> go.Figure:
    """Generic time-series line chart. Handles: empty input, all-null
    values, mismatched-length input, and the single-point case (Plotly
    renders a line with 1 point as just a marker - explicit
    mode='lines+markers' makes that visible instead of an empty-looking
    line)."""
    logger.debug(f"series_line_chart: title={title!r}, n={len(values)}")
    if len(periods) != len(values):
        logger.warning(f"series_line_chart: length mismatch periods={len(periods)} values={len(values)}")
        return _empty_figure(title, f"Veri hatasi: {len(periods)} donem, {len(values)} deger (uyusmuyor)")

    df = pd.DataFrame({"Period": periods, "Deger": values}).dropna(subset=["Deger"])
    if df.empty:
        logger.warning(f"series_line_chart: no non-null data for {title!r}")
        return _empty_figure(title, "Veri yok")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Period"], y=df["Deger"], mode="lines+markers", name=y_label or title))
    fig.update_layout(title=title, xaxis_title="Donem", yaxis_title=y_label, hovermode="x unified")
    return fig


def causality_lag_chart(causality_result: dict) -> go.Figure:
    """Bar chart of lag-correlation strength per lag, from
    causality_analysis()'s own output - highlights the best lag so the
    reader sees which lag the tool's own verdict is based on."""
    lag_correlations = causality_result.get("lag_correlations") or []
    if not lag_correlations:
        return _empty_figure("Lag Korelasyonu", "Veri yok")

    df = pd.DataFrame(lag_correlations)
    best_lag = (causality_result.get("best_lag") or {}).get("lag")
    colors = ["crimson" if row["lag"] == best_lag else "steelblue" for _, row in df.iterrows()]

    fig = go.Figure(go.Bar(x=df["lag"], y=df["correlation"], marker_color=colors))
    fig.update_layout(title="Lag Korelasyonu (kirmizi = en iyi lag)", xaxis_title="Lag (donem)", yaxis_title="Pearson korelasyon")
    return fig


def anomaly_chart(periods: list, values: list, anomaly_result: dict) -> go.Figure:
    """Line chart with anomaly_analysis()'s flagged points highlighted."""
    if len(periods) != len(values):
        return _empty_figure("Anomali Tespiti", f"Veri hatasi: {len(periods)} donem, {len(values)} deger")
    df = pd.DataFrame({"Period": periods, "Deger": values}).dropna(subset=["Deger"])
    if df.empty:
        return _empty_figure("Anomali Tespiti", "Veri yok")

    anomaly_periods = {a["period"] for a in (anomaly_result.get("anomalies") or [])}
    df["is_anomaly"] = df["Period"].isin(anomaly_periods)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Period"], y=df["Deger"], mode="lines", name="seri", line=dict(color="steelblue")))
    flagged = df[df["is_anomaly"]]
    if not flagged.empty:
        fig.add_trace(go.Scatter(x=flagged["Period"], y=flagged["Deger"], mode="markers",
                                  name="anomali", marker=dict(color="crimson", size=11, symbol="x")))
    fig.update_layout(title="Anomali Tespiti (MAD)", xaxis_title="Donem", yaxis_title="Deger", hovermode="x unified")
    return fig


def konut_demo_chart() -> go.Figure:
    """The hackathon brief's own worked example, straight from
    gold_konut_demo: nominal vs CPI-deflated housing credit."""
    if not DB_PATH.exists():
        return _empty_figure("Konut Kredisi", "Gold veritabani yok")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "gold_konut_demo" not in tables:
            return _empty_figure("Konut Kredisi", "gold_konut_demo tablosu yok")
        df = con.execute("SELECT * FROM gold_konut_demo ORDER BY Period").fetchdf()
    finally:
        con.close()

    if df.empty:
        return _empty_figure("Konut Kredisi", "Veri yok")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Period"], y=df["Konut_Kredisi_Milyon_TL"], mode="lines", name="Nominal (milyon TL)"))
    fig.add_trace(go.Scatter(x=df["Period"], y=df["Konut_Kredisi_Reel_Milyon_TL"], mode="lines", name="Reel (2021-01 fiyatlariyla)"))
    fig.update_layout(title="Konut Kredisi: Nominal vs Reel", xaxis_title="Donem", yaxis_title="Milyon TL", hovermode="x unified")
    return fig
