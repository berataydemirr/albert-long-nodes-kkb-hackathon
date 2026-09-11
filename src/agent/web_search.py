"""Web-search capability used by the agent's ``web_search`` tool.

The default provider is DuckDuckGo's HTML endpoint, so a demo works without
an API key. Deployments can opt into Tavily by setting
``WEB_SEARCH_PROVIDER=tavily`` and ``TAVILY_API_KEY``. Only compact,
source-attributed search results are returned; page retrieval belongs to the
separate web-URL tool.
"""

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import TAVILY_API_KEY, WEB_SEARCH_PROVIDER

_DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_QUERY_LENGTH = 500
_MAX_RESULTS = 10
_TIMEOUT_SECONDS = 12
_MAX_TITLE_LENGTH = 300
_MAX_SNIPPET_LENGTH = 800


def _validate_query(query: str, max_results: int) -> tuple[str, int]:
    if not isinstance(query, str):
        raise ValueError("web_search query must be a string.")
    query = query.strip()
    if not query:
        raise ValueError("web_search requires a non-empty query.")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(f"web_search query must be at most {_MAX_QUERY_LENGTH} characters.")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= _MAX_RESULTS:
        raise ValueError(f"max_results must be an integer between 1 and {_MAX_RESULTS}.")
    return query, max_results


def _compact_text(value: str, limit: int) -> str:
    """Bound untrusted provider content before it enters the model context."""
    value = " ".join(value.split())
    return value if len(value) <= limit else f"{value[:limit - 1].rstrip()}…"


def _duckduckgo_target(href: str) -> str:
    """Extract the destination from DuckDuckGo's optional redirect URL."""
    href = href.strip()
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        return unquote(parse_qs(parsed.query).get("uddg", [""])[0])
    return href


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    response = requests.get(
        _DUCKDUCKGO_HTML_URL,
        params={"q": query},
        headers={"User-Agent": "KKB-Agentic-Research/1.0 (+web search)"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in soup.select(".result"):
        link = item.select_one("a.result__a")
        if link is None or not link.get("href"):
            continue
        url = _duckduckgo_target(str(link["href"]))
        if not _is_http_url(url) or url in seen_urls:
            continue
        snippet_node = item.select_one(".result__snippet")
        results.append({
            "title": _compact_text(link.get_text(" ", strip=True), _MAX_TITLE_LENGTH),
            "url": url,
            "snippet": _compact_text(snippet_node.get_text(" ", strip=True), _MAX_SNIPPET_LENGTH) if snippet_node else "",
            "source": urlparse(url).netloc,
        })
        seen_urls.add(url)
        if len(results) == max_results:
            break
    return results


def _tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is required when WEB_SEARCH_PROVIDER=tavily.")
    response = requests.post(
        _TAVILY_SEARCH_URL,
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("results", []):
        url = str(item.get("url", ""))
        if not _is_http_url(url):
            continue
        results.append({
            "title": _compact_text(str(item.get("title", "")), _MAX_TITLE_LENGTH),
            "url": url,
            "snippet": _compact_text(str(item.get("content", "")), _MAX_SNIPPET_LENGTH),
            "source": urlparse(url).netloc,
        })
    return results[:max_results]


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return compact, citation-ready results."""
    query, max_results = _validate_query(query, max_results)
    provider = WEB_SEARCH_PROVIDER.lower().strip()
    try:
        if provider == "duckduckgo":
            results = _duckduckgo_search(query, max_results)
        elif provider == "tavily":
            results = _tavily_search(query, max_results)
        else:
            raise RuntimeError("WEB_SEARCH_PROVIDER must be either 'duckduckgo' or 'tavily'.")
    except requests.RequestException as exc:
        raise RuntimeError(f"Web search request failed: {exc}") from exc
    return {
        "query": query,
        "provider": provider,
        "results": results,
        "result_count": len(results),
        "content_trust": "untrusted_search_results",
        "citation_guidance": "Cite results by title and URL. Search snippets are leads, not verified evidence.",
    }
