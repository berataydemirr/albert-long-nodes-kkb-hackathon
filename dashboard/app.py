"""Faz E: canli dashboard. Calistir:

    streamlit run dashboard/app.py

Iki sekme:
  - "Soru Sor": dogal dilde soru -> MIA agent -> cevap + kaynak gosterimi
    + ilgili seriler icin otomatik grafik + HTML rapor indir.
  - "Genel Bakis": gold katmanindan sabit bir KPI paneli (konut kredisi
    nominal/reel, faiz->kredi nedensellik testi, anomali taramasi,
    saglik/bolgesel veri onizlemesi) - MIA API anahtari olmadan da
    calisir, demo gunu API cevap vermezse bile bos ekran degil.
"""
import sys

import duckdb
import streamlit as st

from config.settings import GOLD_DIR, MIA_API_KEY
from src.report.citations import extract_citations, format_citations_markdown
from src.report.charts import series_line_chart, causality_lag_chart, anomaly_chart, query_series_from_gold, konut_demo_chart
from src.report.export import build_report_html, save_report
from src.agent.tools import anomaly_analysis, causality_analysis
from src.common.logging_setup import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = get_logger("dashboard")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"

st.set_page_config(page_title="KKB Agentic Data Analytics", layout="wide")


@st.cache_data(ttl=3600)
def _load_series_catalog() -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("SELECT DISTINCT Kaynak, Seri_Kod, Seri_Adi, Birim FROM gold_fact_long ORDER BY Kaynak, Seri_Kod").fetchdf()
    finally:
        con.close()
    return df.to_dict("records")


@st.cache_data(ttl=3600)
def _run_causality_demo():
    """Faiz orani -> konut kredisi: hackathon brief'inin kendi vitrin
    ornegi. Cached - Granger testi ucuz degil, her sayfa yenilemesinde
    yeniden hesaplanmasin."""
    faiz = query_series_from_gold("TP.BKR.TRY.18")
    konut = query_series_from_gold("tablo_4.konut_kredisi")
    if faiz.empty or konut.empty or len(faiz) < 17 or len(konut) < 17:
        logger.warning("_run_causality_demo: insufficient data")
        return None
    return causality_analysis(
        faiz["Period"].tolist(), faiz["Deger"].tolist(),
        konut["Period"].tolist(), konut["Deger"].tolist(),
        max_lag=6,
    )


@st.cache_data(ttl=3600)
def _run_anomaly_demo(seri_kod: str):
    df = query_series_from_gold(seri_kod)
    if len(df) < 3:
        return None, df
    return anomaly_analysis(df["Period"].tolist(), df["Deger"].tolist(), threshold=3.0), df


def _render_citation_and_charts(trace: list[dict]):
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
            fig = series_line_chart(df["Period"].tolist(), df["Deger"].tolist(), title=seri_kod)
            figures.append(fig)
            st.plotly_chart(fig, use_container_width=True)
    return citation_result, figures


def render_ask_tab():
    st.subheader("Dogal dilde soru sor")

    if not MIA_API_KEY:
        st.error("MIA_API_KEY .env dosyasinda tanimli degil - bu sekme calismaz. "
                 "(.env.example'a bakin.) 'Genel Bakis' sekmesi API anahtari gerektirmez.")
        return

    question = st.text_input("Soru", placeholder="Konut kredisi faiz orani 2021-2026 arasinda nasil degisti?")
    if st.button("Sor", type="primary") and question:
        with st.spinner("MIA agent calisiyor (birkac tur surebilir)..."):
            try:
                from src.agent.orchestrator import ask
                answer, trace = ask(question, return_trace=True)
            except Exception as exc:
                logger.exception("render_ask_tab: ask() failed")
                st.error(f"Agent hata verdi: {exc}")
                return

        st.markdown("### Cevap")
        st.markdown(answer)

        st.markdown("### Kaynaklar")
        citation_result, figures = _render_citation_and_charts(trace)

        html_report = build_report_html(question, answer, trace, figures=figures, title="KKB Agentic Rapor")
        st.download_button("Raporu HTML olarak indir", data=html_report,
                            file_name="kkb_agentic_rapor.html", mime="text/html")


def render_overview_tab():
    st.subheader("Genel Bakis - Gold Katmani KPI Paneli")

    if not DB_PATH.exists():
        st.error("Gold veritabani bulunamadi. `python -m src.gold.build_gold` calistirin.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(konut_demo_chart(), use_container_width=True)
    with col2:
        causality_result = _run_causality_demo()
        if causality_result:
            st.plotly_chart(causality_lag_chart(causality_result), use_container_width=True)
            sig = causality_result["granger_significant_lags"]
            if sig:
                st.info(f"Granger testi: faiz oraninin gecmisi, lag {sig} icin konut kredisini anlamli sekilde onceden tahmin ediyor (p<0.05). {causality_result['warning']}")
            else:
                st.info(f"Granger testinde anlamli bir lag bulunamadi. {causality_result['warning']}")
        else:
            st.warning("Nedensellik testi icin yeterli veri yok.")

    st.markdown("### Anomali Taramasi")
    catalog = _load_series_catalog()
    series_codes = [c["Seri_Kod"] for c in catalog]
    default_idx = series_codes.index("TP.BISPOLFAIZ.TUR") if "TP.BISPOLFAIZ.TUR" in series_codes else 0
    if series_codes:
        chosen = st.selectbox("Seri sec", series_codes, index=default_idx)
        anomaly_result, df = _run_anomaly_demo(chosen)
        if anomaly_result:
            st.plotly_chart(anomaly_chart(df["Period"].tolist(), df["Deger"].tolist(), anomaly_result), use_container_width=True)
            st.caption(f"{len(anomaly_result['anomalies'])} anomali bulundu (MAD skoru >= 3.0)")
        else:
            st.warning("Bu seri icin anomali taramasi yapilamadi (yetersiz veri).")

    st.markdown("### Seri Katalogu")
    st.dataframe(catalog, use_container_width=True, height=300)


def main():
    st.title("KKB Agentic Data Analytics")
    st.caption("Faz D/E - rapor, kaynak gosterimi ve canli dashboard")
    tab1, tab2 = st.tabs(["Soru Sor", "Genel Bakis"])
    with tab1:
        render_ask_tab()
    with tab2:
        render_overview_tab()


if __name__ == "__main__":
    main()
