"""Deterministic, policy-first routing for the agent's tool loop.

The router classifies intent for policy, prompting, and observability. Every
safe request receives the complete registered tool set so the model can chain
capabilities (for example web search -> URL extraction -> local analysis).
Unsafe requests are refused before the model or DuckDB is reached.
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


def _tool_names(tool_schemas: Iterable[dict]) -> tuple[str, ...]:
    """Extract valid function names from OpenAI-style tool schemas."""
    return tuple(
        schema["function"]["name"]
        for schema in tool_schemas
        if schema.get("type") == "function" and schema.get("function", {}).get("name")
    )


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
    if _URL_RE.search(question):
        if "web_url_extract" in names:
            return RouteDecision(Route.WEB_URL, "Kullanıcı bir URL içeriği istiyor; tüm araçlar zincirlenebilir.", names)
        return RouteDecision(Route.WEB_URL, "URL çıkarma aracı henüz yapılandırılmadı.", available=False)

    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _WEB_SEARCH_PATTERNS):
        if "web_search" in names:
            return RouteDecision(Route.WEB_SEARCH, "Güncel/web kaynağı isteniyor; tüm araçlar zincirlenebilir.", names)
        return RouteDecision(Route.WEB_SEARCH, "Web arama aracı henüz yapılandırılmadı.", available=False)

    return RouteDecision(Route.LOCAL_ANALYTICS, "Tüm kayıtlı araçlar kullanılabilir ve zincirlenebilir.", names)
