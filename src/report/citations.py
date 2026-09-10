"""Faz D1: kaynak gosterimi (source citation).

Turns an orchestrator.ask(return_trace=True) trace into a human-readable
citation list - which Seri_Kod/table a lakehouse_query touched, which
statistical tool ran. Exists so a report can say "this number came from
Seri_Kod X, table Y" instead of asking the reader to trust prose the
model wrote, and so an UNGROUNDED answer (no tool call at all) is
flagged rather than silently presented as if it were.
"""
import json
import re

from src.common.logging_setup import get_logger

logger = get_logger("citations")

SERI_KOD_EQ_RE = re.compile(r"Seri_Kod\s*=\s*'([^']+)'", re.IGNORECASE)
SERI_KOD_IN_RE = re.compile(r"Seri_Kod\s+IN\s*\(([^)]+)\)", re.IGNORECASE)
QUOTED_VALUE_RE = re.compile(r"'([^']+)'")
TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
STAT_TOOLS = {"anomaly_analysis", "change_detection", "correlation_analysis", "causality_analysis", "threshold_filter_compare"}


def _extract_seri_kodlar(sql: str) -> list[str]:
    """Seri_Kod = 'X' AND Seri_Kod IN ('X','Y') both count - a query that
    compares two series (very plausible: 'faiz ile kredi karsilastir')
    typically uses IN, and citations silently missing the second series
    would be worse than missing it loudly."""
    found = set(SERI_KOD_EQ_RE.findall(sql))
    for in_clause in SERI_KOD_IN_RE.findall(sql):
        found.update(QUOTED_VALUE_RE.findall(in_clause))
    return sorted(found)


def extract_citations(trace: list[dict]) -> dict:
    """trace: the list returned by orchestrator.ask(..., return_trace=True).
    Never raises on a malformed trace entry - a citation extractor that
    crashes on bad input would take down report generation for the one
    case (a tool error, odd arguments) where showing *something* matters
    most."""
    logger.info(f"extract_citations: trace has {len(trace)} tool call(s)")

    if not trace:
        logger.warning("extract_citations: empty trace - answer is not grounded in any tool call")
        return {"grounded": False, "citations": [], "tool_failures": [],
                "warning": "Model hicbir tool cagirmadan cevap verdi - bu cevabin veriye dayanip dayanmadigi belirsiz."}

    citations = []
    tool_failures = []
    for entry in trace:
        name = entry.get("name")
        if entry.get("error"):
            tool_failures.append({"tool": name, "error": entry["error"]})
            logger.debug(f"extract_citations: tool failure recorded for {name}")
            continue

        try:
            args = json.loads(entry.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"extract_citations: could not parse arguments for {name}: {exc}")
            args = {}

        if name == "lakehouse_query":
            sql = args.get("sql", "") or ""
            tables = sorted(set(TABLE_RE.findall(sql)))
            series = _extract_seri_kodlar(sql)
            citations.append({"tool": name, "tables": tables, "seri_kodlar": series})
        elif name == "lakehouse_schema":
            continue  # discovery call, not itself a data citation
        elif name in STAT_TOOLS:
            citations.append({"tool": name, "detail": "istatistiksel hesaplama (yerel Python)"})
        elif name:
            citations.append({"tool": name, "detail": "bilinmeyen arac turu"})

    seen, deduped = set(), []
    for c in citations:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    result = {
        "grounded": len(deduped) > 0,
        "citations": deduped,
        "tool_failures": tool_failures,
        "warning": None if deduped else "Tool cagrisi vardi ama gosterilecek bir veri sorgusu/hesaplama bulunamadi.",
    }
    logger.info(f"extract_citations: grounded={result['grounded']}, {len(deduped)} citation(s), {len(tool_failures)} failure(s)")
    return result


def format_citations_markdown(citation_result: dict) -> str:
    """Renders extract_citations()'s output as a short Markdown block for
    embedding under a report's answer text."""
    if not citation_result["grounded"]:
        return f"> Kaynak yok: {citation_result['warning']}"

    lines = ["**Kaynaklar:**"]
    for c in citation_result["citations"]:
        if c["tool"] == "lakehouse_query":
            parts = []
            if c["tables"]:
                parts.append(f"tablo: {', '.join(c['tables'])}")
            if c["seri_kodlar"]:
                parts.append(f"seri: {', '.join(c['seri_kodlar'])}")
            lines.append(f"- Lakehouse sorgusu ({'; '.join(parts) if parts else 'filtre yok'})")
        else:
            lines.append(f"- {c['tool']}: {c.get('detail', '')}")
    if citation_result["tool_failures"]:
        lines.append("")
        lines.append("**Basarisiz tool cagrilari (goz ardi edilmedi, burada gorunuyor):**")
        for f in citation_result["tool_failures"]:
            lines.append(f"- {f['tool']}: {f['error']}")
    return "\n".join(lines)
