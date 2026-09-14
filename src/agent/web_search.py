"""Web-search capability used by the agent's ``web_search`` tool.

Two providers, both usable with no paid third-party API - the project's
open-source requirement rules out a commercial search API (Tavily was used
briefly and removed for exactly this reason; see git history if that
trade-off ever needs revisiting):

  - "duckduckgo" (default): scrapes html.duckduckgo.com directly. No setup,
    but this IP-reputation-blocks automated/cloud traffic - confirmed live,
    it returns a CAPTCHA challenge from this network, not results.
  - "searxng": a self-hosted SearXNG meta-search instance (AGPL-3.0, see
    docker-compose.searxng.yml). It fans a query out to whichever of its
    ~70 backend engines are reachable, so one engine's block does not take
    search down entirely. Requires SEARXNG_BASE_URL to point at a running
    instance with `search.formats: [html, json]` set in its settings.yml -
    the JSON API is off by default.

Only compact, source-attributed search results are returned; page retrieval
belongs to the separate web-URL tool.
"""

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import SEARXNG_BASE_URL, WEB_SEARCH_PROVIDER
from src.common.logging_setup import get_logger

logger = get_logger("web_search")

_DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
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


class SearchProviderUnavailable(RuntimeError):
    """The provider answered, but with a bot challenge or an unparseable
    page rather than results.

    This is separated from "no matches" on purpose. Both used to surface
    as an empty result list, and an empty list reads to the model as "your
    query found nothing" - so it rephrases and calls again, and again. A
    measured run spent 4 web_search calls across 5 rounds that way while
    DuckDuckGo was serving a CAPTCHA to every one of them. Rephrasing
    cannot fix a block, so the model has to be told which case it is in.
    """


def _looks_like_a_bot_challenge(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in (
        "bots use duckduckgo",
        "complete the following challenge",
        "confirm this search was made by a human",
        "unusual traffic",
        "are you a robot",
    ))


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    response = requests.get(
        _DUCKDUCKGO_HTML_URL,
        params={"q": query},
        headers={"User-Agent": "KKB-Agentic-Research/1.0 (+web search)"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if _looks_like_a_bot_challenge(response.text):
        raise SearchProviderUnavailable(
            "DuckDuckGo returned a bot challenge instead of results."
        )
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
    if not results and not soup.select(".result"):
        raise SearchProviderUnavailable(
            "DuckDuckGo returned a page with no result blocks; the endpoint is "
            "blocked or its markup changed."
        )
    return results


def _searxng_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Query a self-hosted SearXNG instance's JSON API.

    Confirmed live against a local instance rather than assumed from docs:
    results carry {url, title, content, engine, engines, score, ...}, and
    the response also carries `unresponsive_engines` - e.g.
    [["duckduckgo", "CAPTCHA"]] was observed live, and results still came
    back from the engines that WERE reachable. That partial-degradation
    behaviour is the whole reason this replaced direct DuckDuckGo scraping:
    one blocked backend engine does not take search down.

    Every failure mode below was reproduced against a real instance rather
    than guessed, because a jury poking at the demo is exactly the
    condition under which an unguarded one surfaces: stopping the
    container mid-session, a slow response, a malformed backend reply.
    All of them raise SearchProviderUnavailable, which web_search() turns
    into a clean "do not retry" signal instead of either a crash or a
    false "no results" that invites a rephrase-and-retry loop.
    """
    try:
        response = requests.get(
            f"{SEARXNG_BASE_URL.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            headers={"User-Agent": "KKB-Agentic-Research/1.0 (+web search)"},
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise SearchProviderUnavailable(
            f"SearXNG did not respond within {_TIMEOUT_SECONDS}s."
        ) from exc
    except requests.ConnectionError as exc:
        # The most common failure in practice: the container is not
        # running. The underlying OS socket error (a raw
        # ConnectionAbortedError with a Windows error code, in testing) is
        # deliberately NOT included in the MESSAGE - it is internal
        # plumbing detail that adds nothing a model or a person reading
        # the trace can act on, only noise. `from exc` still keeps it in
        # the real traceback/logs for debugging.
        raise SearchProviderUnavailable(
            f"Could not reach the SearXNG instance at {SEARXNG_BASE_URL} "
            f"(connection refused/reset) - is the container running? "
            f"`docker compose -f docker-compose.searxng.yml up -d`"
        ) from exc

    if response.status_code == 403:
        raise SearchProviderUnavailable(
            "SearXNG instance rejected the JSON request (HTTP 403) - "
            "search.formats in its settings.yml likely does not include 'json'."
        )
    if response.status_code == 429:
        raise SearchProviderUnavailable(
            "SearXNG instance is rate-limiting requests (HTTP 429)."
        )
    if response.status_code >= 500:
        # A 5xx is the instance itself failing (e.g. its Redis/Valkey cache
        # backend down) - not a code bug on our side and not "no results",
        # so it gets the same clean signal as every other unavailable case
        # rather than a raw HTTPError bubbling up as a generic tool error.
        raise SearchProviderUnavailable(
            f"SearXNG instance returned a server error (HTTP {response.status_code})."
        )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise SearchProviderUnavailable(f"SearXNG response was not valid JSON: {exc}") from exc

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        # A backend response shape change (SearXNG upgrade, a proxy
        # rewriting the body) must not crash the whole agent turn on a
        # .get() call against something that turned out not to be a dict.
        logger.warning(f"_searxng_search: 'results' was {type(raw_results).__name__}, not a list; treating as empty")
        raw_results = []

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            logger.warning(f"_searxng_search: skipping non-dict result entry: {item!r}")
            continue
        url = str(item.get("url", ""))
        if not _is_http_url(url) or url in seen_urls:
            continue
        results.append({
            "title": _compact_text(str(item.get("title", "")), _MAX_TITLE_LENGTH),
            "url": url,
            "snippet": _compact_text(str(item.get("content", "")), _MAX_SNIPPET_LENGTH),
            "source": urlparse(url).netloc,
            # Which backend engine actually answered - useful for a human
            # debugging "why did this result stop coming back" without
            # needing to check the SearXNG container's own logs.
            "engine": str(item.get("engine", "")),
        })
        seen_urls.add(url)
        if len(results) == max_results:
            break

    unresponsive = payload.get("unresponsive_engines")
    if unresponsive:
        # Informational, not fatal: SearXNG still answered from whatever
        # engines WERE reachable. Logged so a degraded-but-working state
        # is visible without digging through the container's own logs.
        logger.info(f"_searxng_search: {len(unresponsive)} engine(s) unresponsive: {unresponsive}")

    return results


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return compact, citation-ready results."""
    query, max_results = _validate_query(query, max_results)
    provider = WEB_SEARCH_PROVIDER.lower().strip()
    try:
        if provider == "duckduckgo":
            results = _duckduckgo_search(query, max_results)
        elif provider == "searxng":
            results = _searxng_search(query, max_results)
        else:
            raise RuntimeError("WEB_SEARCH_PROVIDER must be either 'duckduckgo' or 'searxng'.")
    except SearchProviderUnavailable as exc:
        # Returned, not raised: the model needs to read this and change
        # strategy. Raising would surface as a generic tool error, which
        # reads like a transient fault worth retrying - the opposite of
        # what should happen.
        logger.warning(f"web_search: provider unavailable for {query!r}: {exc}")
        return {
            "query": query,
            "provider": provider,
            "results": [],
            "result_count": 0,
            "provider_available": False,
            "error": str(exc),
            "next_step": (
                "Web arama saglayicisi su an kullanilamiyor. Sorguyu yeniden "
                "yazmak BU DURUMU DEGISTIRMEZ - web_search'u tekrar cagirma. "
                "Yereldeki lakehouse verisiyle cevapla veya guncel bilginin "
                "alinamadigini acikca soyle."
            ),
        }
    except requests.RequestException as exc:
        raise RuntimeError(f"Web search request failed: {exc}") from exc

    logger.info(f"web_search: {query!r} -> {len(results)} result(s) via {provider}")
    payload = {
        "query": query,
        "provider": provider,
        "results": results,
        "result_count": len(results),
        "provider_available": True,
        "content_trust": "untrusted_search_results",
        "citation_guidance": "Cite results by title and URL. Search snippets are leads, not verified evidence.",
    }
    if not results:
        # A genuine zero-match result. One rephrase is reasonable; a third
        # and fourth attempt is the loop this note exists to stop.
        payload["next_step"] = (
            "Bu sorgu icin sonuc yok. En fazla BIR kez farkli anahtar "
            "kelimelerle dene; yine bos donerse bilginin bulunamadigini soyle."
        )
    return payload
