"""Deterministic, policy-first routing for the agent's tool loop.

The LLM still chooses the *specific* local analytical tool.  This module
decides which capability family it is allowed to see.  That keeps a request
for a live web source from silently being answered with stale lakehouse data,
and makes destructive prompt-injection requests fail before they ever reach
the model or DuckDB.

Web routes are intentionally defined now, but marked unavailable until the
corresponding C2/C3 tools are implemented and registered.  This gives later
web-tool work one stable integration point instead of scattering keyword
checks through the orchestrator.
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
    r"\b(ignore|forget|override)\b.{0,60}\b(previous|system|instructions?|rules?)\b",
    r"\b(system prompt|prompt injection|jailbreak)\b",
)
_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
_WEB_SEARCH_PATTERNS = (
    r"\b(web\w*|internet\w*|google\w*|arama\w*|araştır\w*|haber\w*)\b",
    r"\b(güncel|bugün|son durum|latest|current|live|real[ -]?time)\b",
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

    normalized = " ".join(question.casefold().split())
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _UNSAFE_PATTERNS):
        return RouteDecision(
            Route.REFUSE,
            "Veri silme/değiştirme veya talimatları geçersiz kılma isteği işlenmez.",
        )

    names = _tool_names(tool_schemas)
    if _URL_RE.search(question):
        if "web_url_extract" in names:
            return RouteDecision(Route.WEB_URL, "Kullanıcı bir URL içeriği istiyor.", ("web_url_extract",))
        return RouteDecision(Route.WEB_URL, "URL çıkarma aracı henüz yapılandırılmadı.", available=False)

    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _WEB_SEARCH_PATTERNS):
        if "web_search" in names:
            return RouteDecision(Route.WEB_SEARCH, "Güncel/web kaynağı isteniyor.", ("web_search",))
        return RouteDecision(Route.WEB_SEARCH, "Web arama aracı henüz yapılandırılmadı.", available=False)

    return RouteDecision(Route.LOCAL_ANALYTICS, "Yerel lakehouse ve analiz araçları kullanılabilir.", names)
