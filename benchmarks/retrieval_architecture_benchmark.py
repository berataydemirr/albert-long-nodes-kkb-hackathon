"""Compares three retrieval architectures against real TCMB/BDDK/TUIK
targets to decide which one this project should rely on - NOT a
regression test for the currently-wired tool (see web_tool_benchmark.py
for that one). Each architecture is exercised on its own; nothing here
changes which provider src/agent/web_search.py or src/agent/web_url.py
actually use in production.

Architectures under test:
  1. SearXNG   - self-hosted meta-search (docker-compose.searxng.yml),
                 reuses src.agent.web_search's real query function so
                 this measures the actual code path the agent uses.
  2. DuckDuckGo, three variants, for a complete picture:
       - "duckduckgo_search" (the exact package name asked for; PyPI has
         since frozen it in favour of "ddgs" - it still emits a
         RuntimeWarning saying so at call time, captured in the report
         rather than hidden).
       - "ddgs" - the maintained successor package. Included because a
         live check found the legacy package silently returning 0
         results for a real query while ddgs returned 5 real hits for
         the same query - leaving that out would hide the more useful
         option.
       - a raw HTML scrape of html.duckduckgo.com (reuses
         src.agent.web_search's own _duckduckgo_search) - kept as a
         known reference case: this project's own code already
         documents it as CAPTCHA-blocked from this network, which makes
         it a real, reproducible "blocked" resilience case instead of a
         synthetic one.
  3. Playwright + BeautifulSoup - a real headless-browser fetch (so a
     JS-rendered page works, unlike a bare requests.get) with a
     BeautifulSoup text-extraction pass. PDF content is routed to pypdf
     via web_url.py's own _pdf() helper instead of reimplementing PDF
     parsing here - that part is not what differs between architectures,
     it would just be duplicated code. Reuses web_url.py's
     _validate_public_url so this new scraper gets the same
     SSRF/private-network protection as the rest of the project rather
     than a weaker one written from scratch for a benchmark.

Three tasks, per the brief:
  1. Arama performansı  - timed queries against SearXNG and every
     DuckDuckGo variant, scored for both speed and "isabet" (relevance:
     did the results actually include the authoritative domain for the
     query, e.g. tcmb.gov.tr for a TCMB query).
  2. Doğrudan kazıma    - the Playwright+BS4 scraper against two real
     targets, both verified live rather than hand-typed: a genuine
     currently-served EVDS PDF discovered by rendering evds3.tcmb.gov.tr's
     JS-only documents page (its own docs reference an evds2.tcmb.gov.tr
     PDF that turned out to be dead - a 302 to an HTML homepage, not a
     PDF - confirmed while building this benchmark; see EVDS_PDF_URL's
     own comment) and a TUIK inflation page discovered live via a real
     search (see discover_tuik_target()).
  3. Dayanıklılık testi - deliberately hostile targets (timeout, 404, a
     private-network address, a malformed URL, an empty query, the known
     CAPTCHA-blocked DDG scrape) run against whichever architecture(s)
     can meaningfully receive them, checking that each one fails
     *cleanly and identifiably* instead of hanging or crashing.

Usage (from the repo root):
    python -m benchmarks.retrieval_architecture_benchmark
    python -m benchmarks.retrieval_architecture_benchmark --skip-playwright
    python -m benchmarks.retrieval_architecture_benchmark --task 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.agent.web_search import SearchProviderUnavailable, _duckduckgo_search, _searxng_search
from src.agent.web_url import _pdf as _extract_pdf, _validate_public_url
from src.common.logging_setup import get_logger

logger = get_logger("retrieval_architecture_benchmark")

REPORT_PATH = Path(__file__).parent / "retrieval_architecture_benchmark_report.json"

DEFAULT_TIMEOUT_SEC = 15.0
SHORT_TIMEOUT_SEC = 4.0  # deliberately shorter than the httpbin delay/10 resilience case
USER_AGENT = "KKB-Agentic-Research-Benchmark/1.0"


class Outcome(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    BLOCKED = "blocked_or_captcha"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    CONNECTION_ERROR = "connection_error"
    INVALID_INPUT = "invalid_input"
    SSRF_BLOCKED = "ssrf_blocked"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ProbeResult:
    architecture: str
    task: str
    target: str
    outcome: str
    success: bool
    latency_ms: float
    detail: str = ""
    result_count: int = 0
    relevance_score: float | None = None
    # Result URLs (search) - kept as real data on the object rather than
    # embedded in `detail` text, so downstream code (relevance scoring,
    # TUIK URL discovery) never has to re-parse a human-readable string
    # back into structured data.
    sample_urls: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        # Every call site passes an Outcome enum member for readability
        # (Outcome.SUCCESS reads better than the string "success" at the
        # call site); normalize to its plain string value here once so
        # str.format, JSON, and dict-key grouping downstream all see
        # "success" rather than the enum's own repr ("Outcome.SUCCESS").
        if isinstance(self.outcome, Outcome):
            self.outcome = self.outcome.value


def _elapsed_ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000.0, 1)


# --------------------------------------------------------------------------
# Search architectures (Task 1, plus the search-side cases of Task 3)
# --------------------------------------------------------------------------

class SearchArchitecture(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> ProbeResult:
        """Runs one query and returns a fully-classified ProbeResult -
        never raises; every failure mode is caught and mapped to an
        Outcome so the caller can compare architectures uniformly."""


class SearxngArchitecture(SearchArchitecture):
    name = "searxng"

    def search(self, query: str, max_results: int = 5) -> ProbeResult:
        t0 = time.monotonic()
        if not query or not query.strip():
            return ProbeResult(self.name, "search", query, Outcome.INVALID_INPUT, False, _elapsed_ms(t0),
                                detail="Empty query")
        try:
            results = _searxng_search(query, max_results)
        except SearchProviderUnavailable as exc:
            outcome = Outcome.BLOCKED if "captcha" in str(exc).lower() else Outcome.CONNECTION_ERROR
            return ProbeResult(self.name, "search", query, outcome, False, _elapsed_ms(t0), detail=str(exc))
        except requests.Timeout:
            return ProbeResult(self.name, "search", query, Outcome.TIMEOUT, False, _elapsed_ms(t0),
                                detail=f"No response within {DEFAULT_TIMEOUT_SEC}s")
        except requests.RequestException as exc:
            return ProbeResult(self.name, "search", query, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a benchmark must never crash on one architecture's bug
            return ProbeResult(self.name, "search", query, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        latency = _elapsed_ms(t0)
        urls = [r["url"] for r in results]
        outcome = Outcome.SUCCESS if results else Outcome.NO_RESULTS
        return ProbeResult(self.name, "search", query, outcome, bool(results), latency,
                            detail=json.dumps(urls[:3], ensure_ascii=False), result_count=len(results),
                            sample_urls=urls)


class DuckDuckGoRawScrapeArchitecture(SearchArchitecture):
    """The project's own existing direct-scrape implementation
    (src/agent/web_search.py's _duckduckgo_search) - kept as the known
    reference "blocked" case (its own docstring already documents this
    network gets a CAPTCHA challenge here), not as a serious production
    candidate; measuring it live confirms whether that is still true."""
    name = "duckduckgo_raw_scrape"

    def search(self, query: str, max_results: int = 5) -> ProbeResult:
        t0 = time.monotonic()
        if not query or not query.strip():
            return ProbeResult(self.name, "search", query, Outcome.INVALID_INPUT, False, _elapsed_ms(t0),
                                detail="Empty query")
        try:
            results = _duckduckgo_search(query, max_results)
        except SearchProviderUnavailable as exc:
            return ProbeResult(self.name, "search", query, Outcome.BLOCKED, False, _elapsed_ms(t0), detail=str(exc))
        except requests.Timeout:
            return ProbeResult(self.name, "search", query, Outcome.TIMEOUT, False, _elapsed_ms(t0), detail="Timed out")
        except requests.RequestException as exc:
            return ProbeResult(self.name, "search", query, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(self.name, "search", query, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        latency = _elapsed_ms(t0)
        urls = [r["url"] for r in results]
        outcome = Outcome.SUCCESS if results else Outcome.NO_RESULTS
        return ProbeResult(self.name, "search", query, outcome, bool(results), latency,
                            detail=json.dumps(urls[:3], ensure_ascii=False), result_count=len(results),
                            sample_urls=urls)


class DuckDuckGoSearchPackageArchitecture(SearchArchitecture):
    """The legacy "duckduckgo_search" PyPI package - the exact name
    requested. PyPI has since frozen this package in favour of "ddgs";
    importing/using it emits a RuntimeWarning saying so, which is
    captured into the result's detail rather than swallowed, since "the
    package told us it is deprecated" is itself a relevant finding for
    picking a stable long-term dependency."""
    name = "duckduckgo_search_package"

    def search(self, query: str, max_results: int = 5) -> ProbeResult:
        t0 = time.monotonic()
        if not query or not query.strip():
            return ProbeResult(self.name, "search", query, Outcome.INVALID_INPUT, False, _elapsed_ms(t0),
                                detail="Empty query")
        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, TimeoutException
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with DDGS() as client:
                    results = list(client.text(query, max_results=max_results))
            deprecation_note = "; ".join(str(w.message) for w in caught)
        except TimeoutException as exc:
            return ProbeResult(self.name, "search", query, Outcome.TIMEOUT, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except RatelimitException as exc:
            return ProbeResult(self.name, "search", query, Outcome.BLOCKED, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except DuckDuckGoSearchException as exc:
            return ProbeResult(self.name, "search", query, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(self.name, "search", query, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        latency = _elapsed_ms(t0)
        urls = [r.get("href", "") for r in results if r.get("href")]
        outcome = Outcome.SUCCESS if urls else Outcome.NO_RESULTS
        detail = json.dumps(urls[:3], ensure_ascii=False)
        if deprecation_note:
            detail = f"[DEPRECATED: {deprecation_note}] {detail}"
        return ProbeResult(self.name, "search", query, outcome, bool(urls), latency, detail=detail,
                            result_count=len(urls), sample_urls=urls)


class DdgsPackageArchitecture(SearchArchitecture):
    """The maintained successor to duckduckgo_search - same calling
    convention, different package/exception module."""
    name = "ddgs_package"

    def search(self, query: str, max_results: int = 5) -> ProbeResult:
        t0 = time.monotonic()
        if not query or not query.strip():
            return ProbeResult(self.name, "search", query, Outcome.INVALID_INPUT, False, _elapsed_ms(t0),
                                detail="Empty query")
        try:
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
            results = list(DDGS().text(query, max_results=max_results))
        except TimeoutException as exc:
            return ProbeResult(self.name, "search", query, Outcome.TIMEOUT, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except RatelimitException as exc:
            return ProbeResult(self.name, "search", query, Outcome.BLOCKED, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except DDGSException as exc:
            return ProbeResult(self.name, "search", query, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(self.name, "search", query, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        latency = _elapsed_ms(t0)
        urls = [r.get("href", "") for r in results if r.get("href")]
        outcome = Outcome.SUCCESS if urls else Outcome.NO_RESULTS
        return ProbeResult(self.name, "search", query, outcome, bool(urls), latency,
                            detail=json.dumps(urls[:3], ensure_ascii=False), result_count=len(urls),
                            sample_urls=urls)


# --------------------------------------------------------------------------
# Scrape architecture (Task 2, plus the scrape-side cases of Task 3)
# --------------------------------------------------------------------------

class PlaywrightScraper:
    """Headless-Chromium fetch + BeautifulSoup text extraction. One
    browser instance is shared across probes (launching Chromium per call
    would dominate the timing and defeat the point of measuring the
    architecture's own speed); each probe gets its own page for
    isolation. Use as a context manager so the browser is always closed:

        with PlaywrightScraper() as scraper:
            result = scraper.fetch(url)
    """
    name = "playwright_bs4"

    def __init__(self):
        self._playwright = None
        self._browser = None

    def __enter__(self) -> "PlaywrightScraper":
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    def fetch(self, url: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> ProbeResult:
        """Two-step by design, found necessary live: navigating straight to
        a PDF with page.goto() can raise "Download is starting" instead of
        loading it (Chromium treats some PDF responses, e.g. ones served
        with Content-Disposition: inline from certain backends, as a
        download rather than a viewable page) - confirmed against a real
        TCMB EVDS document endpoint. A plain context.request.get() has no
        such ambiguity (it is a raw HTTP fetch, never a navigation), so it
        is always tried first to get real headers/bytes reliably; a page
        is only opened afterwards, and only for HTML, to pick up Playwright's
        actual advantage over bare requests+BeautifulSoup: page.content()
        reflects the DOM AFTER JavaScript has run."""
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        t0 = time.monotonic()
        if self._browser is None:
            raise RuntimeError("PlaywrightScraper must be used as a context manager (`with PlaywrightScraper() as s`).")

        try:
            safe_url = _validate_public_url(url)
        except ValueError as exc:
            outcome = Outcome.SSRF_BLOCKED if "private or reserved" in str(exc) else Outcome.INVALID_INPUT
            return ProbeResult(self.name, "scrape", url, outcome, False, _elapsed_ms(t0), detail=str(exc))

        context = self._browser.new_context(user_agent=USER_AGENT)
        try:
            try:
                api_response = context.request.get(safe_url, timeout=timeout_sec * 1000)
            except PlaywrightTimeoutError as exc:
                return ProbeResult(self.name, "scrape", url, Outcome.TIMEOUT, False, _elapsed_ms(t0),
                                    detail=f"No response within {timeout_sec}s: {exc}")
            except PlaywrightError as exc:
                # Covers DNS failure, connection refused, TLS errors etc. -
                # Playwright's own network errors all surface as this one
                # broad type, distinguished only by message text.
                return ProbeResult(self.name, "scrape", url, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                    detail=f"{type(exc).__name__}: {exc}")

            if api_response.status == 404:
                return ProbeResult(self.name, "scrape", url, Outcome.NOT_FOUND, False, _elapsed_ms(t0),
                                    detail=f"HTTP {api_response.status}")
            if api_response.status >= 400:
                return ProbeResult(self.name, "scrape", url, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                    detail=f"HTTP {api_response.status}")

            content_type = api_response.headers.get("content-type", "").lower()
            body = api_response.body()
            is_pdf = "pdf" in content_type or body[:5] == b"%PDF-"

            try:
                if is_pdf:
                    try:
                        extracted = _extract_pdf(body)
                    except ValueError as exc:
                        return ProbeResult(self.name, "scrape", url, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                            detail=f"PDF parse failed: {exc}")
                    text, kind = extracted["text"], "pdf"
                else:
                    page = context.new_page()
                    try:
                        page.goto(safe_url, wait_until="domcontentloaded", timeout=timeout_sec * 1000)
                        html = page.content()
                    finally:
                        page.close()
                    soup = BeautifulSoup(html, "html.parser")
                    for node in soup(["script", "style", "noscript", "svg", "template"]):
                        node.decompose()
                    text, kind = " ".join(soup.get_text(" ").split()), "html"
            except PlaywrightTimeoutError as exc:
                return ProbeResult(self.name, "scrape", url, Outcome.TIMEOUT, False, _elapsed_ms(t0),
                                    detail=f"No response within {timeout_sec}s: {exc}")
            except PlaywrightError as exc:
                return ProbeResult(self.name, "scrape", url, Outcome.CONNECTION_ERROR, False, _elapsed_ms(t0),
                                    detail=f"{type(exc).__name__}: {exc}")

            latency = _elapsed_ms(t0)
            success = len(text) > 20
            return ProbeResult(self.name, "scrape", url, Outcome.SUCCESS if success else Outcome.NO_RESULTS,
                                success, latency, detail=f"kind={kind}, chars={len(text)}: {text[:200]!r}",
                                result_count=len(text))
        except Exception as exc:  # noqa: BLE001 - extraction bugs must not crash the whole benchmark run
            return ProbeResult(self.name, "scrape", url, Outcome.UNKNOWN_ERROR, False, _elapsed_ms(t0),
                                detail=f"{type(exc).__name__}: {exc}")
        finally:
            context.close()


# --------------------------------------------------------------------------
# Task 1: search performance + relevance ("isabet")
# --------------------------------------------------------------------------

SEARCH_QUERIES = [
    {"query": "TCMB güncel politika faizi", "relevant_domains": ("tcmb.gov.tr",)},
    {"query": "BDDK takipteki krediler raporu", "relevant_domains": ("bddk.org.tr",)},
]


def _relevance(urls: list[str], relevant_domains: tuple[str, ...]) -> float:
    if not urls:
        return 0.0
    hits = sum(1 for u in urls if any(d in urlparse(u).netloc for d in relevant_domains))
    return round(hits / len(urls), 3)


def run_task1_search_performance(architectures: list[SearchArchitecture]) -> list[ProbeResult]:
    logger.info("=" * 70)
    logger.info("TASK 1: ARAMA PERFORMANSI")
    logger.info("=" * 70)
    results: list[ProbeResult] = []
    for spec in SEARCH_QUERIES:
        for arch in architectures:
            result = arch.search(spec["query"], max_results=5)
            if result.success:
                result.relevance_score = _relevance(result.sample_urls, spec["relevant_domains"])
            logger.info(f"  [{result.outcome:20s}] {arch.name:28s} {spec['query']!r:40s} "
                        f"{result.latency_ms:7.1f}ms relevance={result.relevance_score}")
            results.append(result)
    return results


# --------------------------------------------------------------------------
# Task 2: direct scraping of real, hard targets
# --------------------------------------------------------------------------

# A URL already referenced in this project's own docs
# (docs/progressive_finance.md), not invented for this benchmark.
# docs/progressive_finance.md references an EVDS PDF at evds2.tcmb.gov.tr -
# that link is now DEAD (confirmed live: it 302-redirects to
# evds3.tcmb.gov.tr's HTML homepage, not a PDF - the site migrated from
# evds2 to evds3). Rather than keep a known-stale URL in a benchmark whose
# whole point is measuring real retrieval, this one was found live: the
# JS-rendered https://evds3.tcmb.gov.tr/dokumanlar page (its links only
# exist post-JavaScript, a genuine BeautifulSoup-alone dead end and the
# reason this project reaches for Playwright at all) lists this real,
# currently-served PDF (confirmed: HTTP 200, content-type application/pdf).
EVDS_PDF_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/documents/showDocument?docId=18"


def discover_tuik_target(searxng: SearxngArchitecture) -> str | None:
    """Finds a real, currently-live TUIK inflation page via search instead
    of hand-typing a URL that may not exist or may have moved - TUIK
    restructures its bulletin URLs by date/id, so a guessed URL here would
    likely be wrong or stale. Picks the first result whose domain is
    actually tuik.gov.tr."""
    result = searxng.search("TÜİK enflasyon bülteni", max_results=8)
    if not result.success:
        logger.warning("discover_tuik_target: SearXNG probe failed, cannot discover a live TUIK URL")
        return None
    for url in result.sample_urls:
        if "tuik.gov.tr" in urlparse(url).netloc:
            logger.info(f"discover_tuik_target: using {url!r} (found live via search)")
            return url
    return None


def run_task2_direct_scraping(scraper: PlaywrightScraper, tuik_url: str | None) -> list[ProbeResult]:
    logger.info("=" * 70)
    logger.info("TASK 2: DOĞRUDAN KAZIMA")
    logger.info("=" * 70)
    targets = [(EVDS_PDF_URL, "TCMB EVDS Kullanım Şartları (PDF)")]
    if tuik_url:
        targets.append((tuik_url, "TÜİK enflasyon sayfası (canlı keşfedildi)"))
    else:
        logger.warning("run_task2_direct_scraping: no TUIK target discovered, skipping that probe")

    results = []
    for url, label in targets:
        result = scraper.fetch(url)
        logger.info(f"  [{result.outcome:20s}] {label:40s} {result.latency_ms:8.1f}ms {result.detail[:100]}")
        results.append(result)
    return results


# --------------------------------------------------------------------------
# Task 3: resilience under hostile conditions
# --------------------------------------------------------------------------

def run_task3_resilience(search_architectures: list[SearchArchitecture],
                          scraper: PlaywrightScraper | None) -> list[ProbeResult]:
    logger.info("=" * 70)
    logger.info("TASK 3: DAYANIKLILIK TESTİ")
    logger.info("=" * 70)
    results: list[ProbeResult] = []

    # Empty query - every search architecture must reject this cleanly,
    # not hang or crash on an unbounded/empty request.
    for arch in search_architectures:
        result = arch.search("")
        logger.info(f"  [{result.outcome:20s}] {arch.name:28s} empty-query check {result.latency_ms:7.1f}ms")
        results.append(result)

    # The known CAPTCHA-blocked case - already exercised as part of Task 1
    # for the raw-scrape architecture, but re-run here explicitly as a
    # named resilience case so it shows up in the resilience summary too.
    ddg_raw = DuckDuckGoRawScrapeArchitecture()
    result = ddg_raw.search("TCMB politika faizi", max_results=3)
    result.task = "resilience"
    logger.info(f"  [{result.outcome:20s}] {ddg_raw.name:28s} known-captcha-block check {result.latency_ms:7.1f}ms")
    results.append(result)

    # Scrape-side hostile targets - skipped when the browser was not
    # started (--skip-playwright), rather than crashing on a None scraper.
    if scraper is None:
        logger.warning("run_task3_resilience: no Playwright scraper available, skipping scrape-side cases")
        return results

    scrape_cases = [
        ("https://httpbin.org/delay/10", "timeout (short client budget vs a 10s-delay endpoint)", SHORT_TIMEOUT_SEC),
        ("https://httpbin.org/status/404", "not-found", DEFAULT_TIMEOUT_SEC),
        ("http://127.0.0.1:65535/", "private-network target (must be refused, not attempted)", DEFAULT_TIMEOUT_SEC),
        ("not-a-url-at-all", "malformed URL", DEFAULT_TIMEOUT_SEC),
    ]
    for url, label, timeout in scrape_cases:
        result = scraper.fetch(url, timeout_sec=timeout)
        result.task = "resilience"
        logger.info(f"  [{result.outcome:20s}] playwright_bs4 {label:48s} {result.latency_ms:8.1f}ms {result.detail[:80]}")
        results.append(result)

    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _summarize(results: list[ProbeResult]) -> dict:
    by_arch: dict[str, list[ProbeResult]] = {}
    for r in results:
        by_arch.setdefault(r.architecture, []).append(r)

    summary = {}
    for arch, probes in by_arch.items():
        latencies = [p.latency_ms for p in probes]
        outcomes: dict[str, int] = {}
        for p in probes:
            outcomes[p.outcome] = outcomes.get(p.outcome, 0) + 1
        relevances = [p.relevance_score for p in probes if p.relevance_score is not None]
        summary[arch] = {
            "probe_count": len(probes),
            "success_count": sum(1 for p in probes if p.success),
            "success_rate": round(sum(1 for p in probes if p.success) / len(probes), 3),
            "avg_latency_ms": round(statistics.mean(latencies), 1),
            "median_latency_ms": round(statistics.median(latencies), 1),
            "avg_relevance": round(statistics.mean(relevances), 3) if relevances else None,
            "outcome_breakdown": outcomes,
        }
    return summary


def _print_summary(summary: dict) -> None:
    print("=" * 100)
    print("ÖZET (mimari başına)")
    print("=" * 100)
    header = f"{'Mimari':30s} {'Prob':>5s} {'Başarı':>8s} {'Ort.Gecikme':>12s} {'Ort.İsabet':>11s}  Hata Dağılımı"
    print(header)
    print("-" * 100)
    for arch, s in sorted(summary.items()):
        relevance = f"{s['avg_relevance']:.2f}" if s["avg_relevance"] is not None else "n/a"
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(s["outcome_breakdown"].items()))
        print(f"{arch:30s} {s['probe_count']:5d} {s['success_rate']*100:7.1f}% "
              f"{s['avg_latency_ms']:10.1f}ms {relevance:>11s}  {breakdown}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SearXNG / DuckDuckGo variants / Playwright+BS4")
    parser.add_argument("--skip-playwright", action="store_true", help="Skip Task 2/3's browser-based probes")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=None, help="Run only one task (default: all)")
    args = parser.parse_args()

    search_architectures: list[SearchArchitecture] = [
        SearxngArchitecture(), DuckDuckGoSearchPackageArchitecture(), DdgsPackageArchitecture(),
    ]

    all_results: list[ProbeResult] = []
    want_task2 = args.task in (None, 2)
    want_task3 = args.task in (None, 3)

    if args.task in (None, 1):
        all_results += run_task1_search_performance(search_architectures)

    if (want_task2 or want_task3) and args.skip_playwright:
        if want_task2:
            logger.warning("main: --skip-playwright set, Task 2 (Playwright-only) is entirely skipped")
        if want_task3:
            logger.warning("main: --skip-playwright set, Task 3 runs only its search-side cases")
            all_results += run_task3_resilience(search_architectures, scraper=None)
    elif want_task2 or want_task3:
        searxng = SearxngArchitecture()
        with PlaywrightScraper() as scraper:
            if want_task2:
                tuik_url = discover_tuik_target(searxng)
                all_results += run_task2_direct_scraping(scraper, tuik_url)
            if want_task3:
                all_results += run_task3_resilience(search_architectures, scraper)

    summary = _summarize(all_results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_count": len(all_results),
        "summary": summary,
        "probes": [asdict(r) for r in all_results],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(summary)
    print(f"Tam rapor: {REPORT_PATH}")


if __name__ == "__main__":
    main()
