"""
KKB Agentic Data Analytics — Streamlit Dashboard

Tabs:
  - Overview: Gold-layer KPI panel (no API key required, safe for demo day).
  - Ask: Natural-language question -> MIA agent -> answer + citations + chart + HTML export.

Run:
    streamlit run dashboard/app.py
"""
import sys
import duckdb
import streamlit as st

from config.settings import GOLD_DIR, MIA_API_KEY
from src.report.citations import extract_citations, format_citations_markdown
from src.report.charts import (
    series_line_chart, causality_lag_chart, anomaly_chart,
    query_series_from_gold, konut_demo_chart,
)
from src.report.export import build_report_html
from src.agent.tools import anomaly_analysis, causality_analysis
from src.common.logging_setup import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = get_logger("dashboard")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KKB Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (minimal, clean) ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Inter-like system font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* KPI cards */
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .kpi-label { font-size: 0.78rem; font-weight: 600; color: #64748b;
                 text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1; }
    .kpi-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem; }

    /* Section headers */
    .section-header {
        font-size: 1rem; font-weight: 700; color: #0f172a;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem;
        margin-bottom: 1rem; margin-top: 1.5rem;
    }

    /* Answer box */
    .answer-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #166534;
    }

    /* Chip badge */
    .chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        margin: 0.1rem;
    }

    /* Tab overrides */
    [data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_series_catalog() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim "
            "FROM gold_fact_long ORDER BY Kaynak, Seri_Kod"
        ).fetchdf()
    finally:
        con.close()
    return df.to_dict("records")


@st.cache_data(ttl=3600)
def _run_causality_demo():
    faiz  = query_series_from_gold("TP.BKR.TRY.18")
    konut = query_series_from_gold("tablo_4.konut_kredisi")
    if faiz.empty or konut.empty or len(faiz) < 17 or len(konut) < 17:
        logger.warning("_run_causality_demo: insufficient data")
        return None
    return causality_analysis(
        faiz["Period"].tolist(),  faiz["Deger"].tolist(),
        konut["Period"].tolist(), konut["Deger"].tolist(),
        max_lag=6,
    )


@st.cache_data(ttl=3600)
def _run_anomaly_demo(seri_kod: str):
    df = query_series_from_gold(seri_kod)
    if len(df) < 3:
        return None, df
    return anomaly_analysis(
        df["Period"].tolist(), df["Deger"].tolist(), threshold=3.0
    ), df


# ── Helper widgets ─────────────────────────────────────────────────────────────
def _kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
            </div>""",
        unsafe_allow_html=True,
    )


def _section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _render_citations(trace: list[dict]):
    citation_result = extract_citations(trace)
    st.markdown(format_citations_markdown(citation_result))
    figures = []
    for c in citation_result["citations"]:
        if c["tool"] != "lakehouse_query":
            continue
        for seri_kod in c.get("seri_kodlar", []):
            df = query_series_from_gold(seri_kod)
            if df.empty:
                continue
            fig = series_line_chart(
                df["Period"].tolist(), df["Deger"].tolist(), title=seri_kod
            )
            figures.append(fig)
            st.plotly_chart(fig, use_container_width=True)
    return citation_result, figures


# ── Tab: Overview ──────────────────────────────────────────────────────────────
def render_overview():
    if not DB_PATH.exists():
        st.error(
            "Gold database not found. Run `python -m src.gold.build_gold` first.",
            icon="🚫",
        )
        return

    catalog = _load_series_catalog()

    # ── KPI row ────────────────────────────────────────────────────────────────
    _section("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Series in Gold", str(len(catalog)), "across all sources")
    with c2:
        sources = len({r["Kaynak"] for r in catalog})
        _kpi("Data Sources", str(sources), "EVDS · BDDK · TUIK")
    with c3:
        _kpi("Coverage", "2021–2026", "monthly granularity")
    with c4:
        _kpi("Database", "DuckDB", "gold_fact_long")

    # ── Housing credit chart ───────────────────────────────────────────────────
    _section("Housing Credit — Nominal vs Real")
    try:
        st.plotly_chart(konut_demo_chart(), use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart unavailable: {exc}")

    # ── Causality test ─────────────────────────────────────────────────────────
    _section("Granger Causality: Policy Rate → Housing Credit")
    causality_result = _run_causality_demo()
    if causality_result:
        st.plotly_chart(causality_lag_chart(causality_result), use_container_width=True)
        sig = causality_result["granger_significant_lags"]
        if sig:
            st.success(
                f"Significant at lags {sig} (p < 0.05). {causality_result['warning']}"
            )
        else:
            st.info(f"No significant lag found. {causality_result['warning']}")
    else:
        st.warning("Insufficient data for causality test.")

    # ── Anomaly scanner ────────────────────────────────────────────────────────
    _section("Anomaly Scanner")
    series_codes = [c["Seri_Kod"] for c in catalog]
    if series_codes:
        default_idx = (
            series_codes.index("TP.BISPOLFAIZ.TUR")
            if "TP.BISPOLFAIZ.TUR" in series_codes
            else 0
        )
        chosen = st.selectbox("Select series", series_codes, index=default_idx)
        anomaly_result, df = _run_anomaly_demo(chosen)
        if anomaly_result:
            st.plotly_chart(
                anomaly_chart(
                    df["Period"].tolist(), df["Deger"].tolist(), anomaly_result
                ),
                use_container_width=True,
            )
            n = len(anomaly_result["anomalies"])
            st.caption(f"{n} anomal{'y' if n == 1 else 'ies'} detected (MAD score ≥ 3.0)")
        else:
            st.warning("Insufficient data for anomaly scan on this series.")

    # ── Series catalog ─────────────────────────────────────────────────────────
    _section("Series Catalog")
    st.dataframe(catalog, use_container_width=True, height=280)


# ── Tab: Ask ───────────────────────────────────────────────────────────────────
def render_ask():
    if not MIA_API_KEY:
        st.error(
            "**MIA_API_KEY** is not set in `.env`. "
            "This tab requires a valid key. The Overview tab works without one.",
            icon="🔑",
        )
        return

    st.markdown(
        "Ask a question in plain language. The agent will query the gold lakehouse, "
        "run statistical tools, and cite every number it uses."
    )

    question = st.text_input(
        "Question",
        placeholder="How did the housing credit interest rate change between 2021 and 2026?",
        label_visibility="collapsed",
    )

    col_btn, col_spacer = st.columns([1, 5])
    with col_btn:
        submitted = st.button("Ask →", type="primary", use_container_width=True)

    if submitted and question:
        with st.spinner("Agent is working…"):
            try:
                from src.agent.orchestrator import ask
                answer, trace = ask(question, return_trace=True)
            except Exception as exc:
                logger.exception("render_ask: ask() failed")
                st.error(f"Agent error: {exc}")
                return

        # Answer
        _section("Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True
        )

        # Citations & charts
        _section("Sources & Charts")
        citation_result, figures = _render_citations(trace)

        # Download
        html_report = build_report_html(
            question, answer, trace, figures=figures, title="KKB Agentic Report"
        )
        st.download_button(
            "⬇  Download HTML report",
            data=html_report,
            file_name="kkb_report.html",
            mime="text/html",
        )


# ── App shell ──────────────────────────────────────────────────────────────────
def main():
    # Compact header
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.markdown("## 📊")
    with col_title:
        st.markdown("## KKB Agentic Data Analytics")
        st.caption("Bronze → Silver → Gold lakehouse · BDDK · EVDS · TUIK")

    st.markdown("---")

    tab_overview, tab_ask = st.tabs(["📈  Overview", "💬  Ask a question"])
    with tab_overview:
        render_overview()
    with tab_ask:
        render_ask()


if __name__ == "__main__":
    main()
