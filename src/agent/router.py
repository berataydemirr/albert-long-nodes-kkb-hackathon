"""Deterministic, policy-first routing for the agent's tool loop.

The roadmap assigns one narrow use case to each tool.  The router therefore
exposes only the capabilities required by the question, while still allowing
mixed questions to receive the union of their required tools.  This keeps a
small model from wandering into unrelated, expensive tool chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


class Route(str, Enum):
    REFUSE = "refuse"
    LOCAL_ANALYTICS = "local_analytics"
    WEB_SEARCH = "web_search"
    WEB_URL = "web_url"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    allowed_tool_names: tuple[str, ...] = ()
    available: bool = True


# Intentional small deny-list: it is an early safety gate, not the sole SQL
# guardrail. lakehouse_query keeps its own strict validation as defense in
# depth.
_UNSAFE_PATTERNS = (
    r"\b(drop|delete|truncate|alter|insert|update|create)\s+(table|database|schema|from|into)\b",
    r"\bupdate\s+(?:\w+\.)?\w+\s+set\b",
    r"\b(ignore|forget|override|unut|atla)\b.{0,60}\b(previous|system|instructions?|rules?|onceki|talimatlar?(?:ı|i|in|ın)?|kurallar?(?:ı|i|in|ın)?)\b",
    r"\b(previous|system|instructions?|rules?|onceki|talimatlar?(?:ı|i|in|ın)?|kurallar?(?:ı|i|in|ın)?)\b.{0,60}\b(ignore|forget|override|unut|atla)\b",
    r"\b(system prompt|prompt injection|jailbreak)\b",
)
_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
_WEB_SEARCH_PATTERNS = (
    r"\b(web\w*|internet\w*|google\w*|arama\w*|araştır\w*|haber\w*)\b",
    r"\b(güncel\w*|bugün\w*|son durum|latest|current|live|real[ -]?time)\b",
)

_SCHEMA_PATTERNS = (
    r"\b(şema\w*|schema|tablo\w*|kolon\w*|sütun\w*|seri(?:ler)?\w*\s+(?:listele|göster)|hangi\s+(?:tablo|seri))\b",
)
_ANOMALY_PATTERNS = (
    r"\b(aykırı\w*|anomali\w*|anomal\w*|outlier\w*|z[ -]?score|iqr)\b",
)
_CHANGE_PATTERNS = (
    r"\b(değişim\w*|degisim\w*|trend\s+kırıl\w*|kırılma\w*|en\s+büyük\w*\s+(?:aylık\s+)?(?:değişim|artış|düşüş)|change[ -]?point)\b",
)
_CORRELATION_PATTERNS = (
    r"\b(korelasyon\w*|correlation\w*|ilişki\w*)\b",
)
_CAUSALITY_PATTERNS = (
    r"\b(nedensel\w*|nedensellik\w*|granger\w*|öngör\w*|etki\w*)\b",
)
_THRESHOLD_PATTERNS = (
    r"(?:%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%)\s*(?:-|–|—|ile|ve)\s*(?:%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%)\s*(?:arası|arasındayken|aralığında)",
    r"\b(eşik\w*|threshold\w*|filtre\w*)\b",
)
_LOCAL_DATA_PATTERNS = (
    r"\b(lakehouse|gold_|bddk|evds|tuik|kredi\w*|faiz\w*|konut\w*|fiyat\w*|hacim\w*|seri\w*|dönem\w*|veri\w*)\b",
)


def _tool_names(tool_schemas: Iterable[dict]) -> tuple[str, ...]:
    """Extract valid function names from OpenAI-style tool schemas."""
    return tuple(
        schema["function"]["name"]
        for schema in tool_schemas
        if schema.get("type") == "function" and schema.get("function", {}).get("name")
    )


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _available_in_order(all_names: tuple[str, ...], wanted: set[str]) -> tuple[str, ...]:
    """Keep registry order so prompts and traces remain deterministic."""
    return tuple(name for name in all_names if name in wanted)


def _local_capabilities(text: str) -> tuple[set[str], list[str]]:
    """Map roadmap C1/C4/C5/C6 language to the minimum local tool set."""
    wanted: set[str] = set()
    reasons: list[str] = []

    schema_only = _matches_any(text, _SCHEMA_PATTERNS)
    anomaly = _matches_any(text, _ANOMALY_PATTERNS)
    change = _matches_any(text, _CHANGE_PATTERNS)
    causality = _matches_any(text, _CAUSALITY_PATTERNS)
    threshold = _matches_any(text, _THRESHOLD_PATTERNS)
    correlation = _matches_any(text, _CORRELATION_PATTERNS) and not causality
    local_data = _matches_any(text, _LOCAL_DATA_PATTERNS)

    if schema_only and not any((anomaly, change, correlation, causality, threshold)):
        wanted.add("lakehouse_schema")
        reasons.append("C1 şema/seri kataloğu")
    elif local_data or any((anomaly, change, correlation, causality, threshold)):
        wanted.update(("lakehouse_schema", "lakehouse_query"))
        reasons.append("C1 yerel veri keşfi ve sorgusu")

    if anomaly:
        wanted.add("anomaly_analysis")
        reasons.append("C4 aykırı değer analizi")
    if change:
        wanted.add("change_detection")
        reasons.append("C6 değişim tespiti")
    if threshold:
        wanted.add("threshold_filter_compare")
        reasons.append("C5 eşik aralığı karşılaştırması")
    elif causality:
        wanted.add("causality_analysis")
        reasons.append("C5 lag/Granger analizi")
    elif correlation:
        wanted.add("correlation_analysis")
        reasons.append("C5 korelasyon analizi")

    return wanted, reasons


def route_question(question: str, tool_schemas: Iterable[dict]) -> RouteDecision:
    """Classify a user request without network or model calls.

    Precedence is deliberate: unsafe input always wins; a supplied URL is
    more specific than a generic web-search phrase; all remaining questions
    stay in the local analytical capability family.
    """
    if not isinstance(question, str) or not question.strip():
        return RouteDecision(Route.REFUSE, "Boş veya geçersiz soru.")

    # Turkish capital İ casefolds to ``i`` plus a combining dot.  Normalize
    # only that sequence: general accent stripping would turn ``ü`` into
    # ``u`` and break Turkish keywords such as ``güncel``.
    normalized = question.casefold().replace("i\u0307", "i")
    normalized = " ".join(normalized.split())
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _UNSAFE_PATTERNS):
        return RouteDecision(
            Route.REFUSE,
            "Veri silme/değiştirme veya talimatları geçersiz kılma isteği işlenmez.",
        )

    names = _tool_names(tool_schemas)
    wanted, reasons = _local_capabilities(normalized)
    has_url = bool(_URL_RE.search(question))
    wants_web = _matches_any(normalized, _WEB_SEARCH_PATTERNS)
    explicit_local_work = _matches_any(
        normalized,
        (r"\b(yerel\w*|lakehouse|gold_|karşılaştır\w*|kıyasla\w*|analiz\w*)\b",),
    ) or any(name in wanted for name in (
        "anomaly_analysis", "change_detection", "correlation_analysis",
        "causality_analysis", "threshold_filter_compare",
    ))

    # A current rate or a supplied URL may mention domain words such as
    # "faiz" without asking for local data. Keep C1 only for an explicit
    # local/analytical part of a mixed request.
    if (has_url or wants_web) and not explicit_local_work:
        wanted.difference_update(("lakehouse_schema", "lakehouse_query"))
        reasons = [reason for reason in reasons if not reason.startswith("C1")]

    if has_url:
        wanted.add("web_url_extract")
        reasons.insert(0, "C3 verilen URL içeriğini çıkarma")
        # A URL is already a concrete source; web_search is only necessary
        # when the user explicitly asks for a broader search as well.
        if wants_web and _matches_any(normalized, (r"\b(ara\w*|search\w*|google\w*|araştır\w*)\b",)):
            wanted.add("web_search")
            reasons.insert(0, "C2 ek web araması")
        selected = _available_in_order(names, wanted)
        if "web_url_extract" in selected:
            return RouteDecision(Route.WEB_URL, "; ".join(reasons) + ".", selected)
        return RouteDecision(Route.WEB_URL, "URL çıkarma aracı henüz yapılandırılmadı.", available=False)

    if wants_web:
        wanted.add("web_search")
        reasons.insert(0, "C2 güncel/harici bilgi araması")
        # Search results are leads. URL extraction lets the model verify a
        # selected source, as required by the system prompt.
        wanted.add("web_url_extract")
        selected = _available_in_order(names, wanted)
        if "web_search" in selected:
            return RouteDecision(Route.WEB_SEARCH, "; ".join(reasons) + ".", selected)
        return RouteDecision(Route.WEB_SEARCH, "Web arama aracı henüz yapılandırılmadı.", available=False)

    if not wanted:
        wanted.update(("lakehouse_schema", "lakehouse_query"))
        reasons.append("C1 yerel veri keşfi ve sorgusu")
    selected = _available_in_order(names, wanted)
    return RouteDecision(Route.LOCAL_ANALYTICS, "; ".join(reasons) + ".", selected)
