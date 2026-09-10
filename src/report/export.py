"""Faz D2: soru + cevap + kaynaklar + grafikleri tek bir self-contained
HTML rapora birlestirir (data/reports/ altina yazilir). PDF degil -
tarayicinin "Yazdir -> PDF" ozelligi HTML'den PDF'e gecisi zaten cozer,
ekstra bir PDF kutuphanesi eklemeye gerek birakmadan.
"""
import datetime as dt
import html as html_lib
from pathlib import Path

import plotly.graph_objects as go

from config.settings import PROJECT_ROOT
from src.report.citations import extract_citations, format_citations_markdown
from src.common.logging_setup import get_logger

logger = get_logger("report_export")

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .answer {{ background: #f7f7f7; border-left: 4px solid #444; padding: 16px 20px; white-space: pre-wrap; margin-bottom: 20px; }}
  .citations {{ background: #fff8e6; border-left: 4px solid #c9a227; padding: 12px 20px; white-space: pre-wrap; font-size: 14px; margin-bottom: 30px; }}
  .chart {{ margin-bottom: 30px; }}
  .warning {{ color: #a83232; font-weight: bold; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Olusturulma: {generated_at}</div>
<h2>Soru</h2>
<div class="answer">{question}</div>
<h2>Cevap</h2>
<div class="answer">{answer}</div>
<h2>Kaynaklar</h2>
<div class="citations">{citations}</div>
{charts_html}
</body>
</html>
"""


def build_report_html(question: str, answer: str, trace: list[dict], figures: list[go.Figure] | None = None, title: str = "KKB Agentic Rapor") -> str:
    """Returns the report as an HTML string - doesn't write a file itself
    (build_and_save_report() does that), so a caller (e.g. the Streamlit
    dashboard) can embed the same HTML inline without a disk round-trip."""
    logger.info(f"build_report_html: question={question!r}, n_figures={len(figures or [])}")

    citation_result = extract_citations(trace)
    citations_md = format_citations_markdown(citation_result)
    if not citation_result["grounded"]:
        citations_md = f'<span class="warning">{html_lib.escape(citations_md)}</span>'
    else:
        citations_md = html_lib.escape(citations_md)

    charts_html = ""
    for i, fig in enumerate(figures or []):
        include_js = "cdn" if i == 0 else False  # plotly.js already loaded via the <script> tag above; avoid re-embedding it per chart
        charts_html += f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart-{i}")}</div>\n'

    return _TEMPLATE.format(
        title=html_lib.escape(title),
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        question=html_lib.escape(question or "(soru bos)"),
        answer=html_lib.escape(answer or "(cevap bos)"),
        citations=citations_md,
        charts_html=charts_html,
    )


def save_report(html_content: str, filename: str | None = None) -> Path:
    """Writes the report to data/reports/<filename>.html - filename
    defaults to a timestamp so repeated runs never silently overwrite
    each other."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    if not filename.endswith(".html"):
        filename += ".html"
    out_path = REPORTS_DIR / filename
    out_path.write_text(html_content, encoding="utf-8")
    logger.info(f"save_report: written to {out_path} ({len(html_content)} chars)")
    return out_path
