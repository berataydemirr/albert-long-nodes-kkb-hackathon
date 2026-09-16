"""Regression benchmark for the web tools (web_search, web_url_extract)
and for whether the agent actually USES them correctly instead of
hallucinating a plausible-sounding answer from parametric memory.

Built after a real incident this session: SearXNG (the local search
backend web_search depends on) was down for most of a working session,
and separately the model was found confidently stating WRONG facts for
domain-specific terminology (e.g. "KMH = Kredi Karti Harcamasi" - it
actually means "Kredili Mevduat Hesabi") instead of checking. Two
different failure classes, both covered here:

  1. Provider-level checks (_test_search_provider_reachable,
     _test_web_search_returns_real_results, _test_web_url_extract): is
     the tool ITSELF working right now - a fast, cheap, no-LLM check that
     should run before blaming the agent for "not searching".

  2. Agent-level questions (AGENT_QUESTIONS): does the agent actually
     call web_search/web_url_extract when a question needs it, and does
     it get the SPECIFIC facts right that were found wrong live (KMH,
     TR7's province list) - regression coverage for the system-prompt
     fix in orchestrator.py's SYSTEM_PROMPT_BASE (the "kisaltma/kod
     anlami" paragraph) telling the model to verify terminology against
     source metadata or web_search rather than its own general
     knowledge.

Run with:
    python -m benchmarks.web_tool_benchmark
    python -m benchmarks.web_tool_benchmark --skip-agent   # provider checks only, fast, no LLM cost
"""
import argparse
import json
import time
from pathlib import Path

from src.agent.orchestrator import ask
from src.agent.web_search import web_search
from src.agent.web_url import web_url_extract
from src.common.logging_setup import get_logger

logger = get_logger("web_tool_benchmark")

REPORT_PATH = Path(__file__).parent / "web_tool_benchmark_report.json"

# A real, stable TCMB page - if this stops being fetchable, the internet
# path itself (not just the search index) is the problem.
_KNOWN_URL = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Para+Politikasi/PPK/2026"


def _test_search_provider_reachable() -> dict:
    """Direct, no-LLM call - isolates 'is the provider up' from 'did the
    agent choose to use it'."""
    t0 = time.monotonic()
    try:
        result = web_search("TCMB politika faizi", max_results=3)
        ok = bool(result.get("provider_available")) and result.get("result_count", 0) > 0
        return {
            "name": "search_provider_reachable", "ok": ok,
            "provider": result.get("provider"), "result_count": result.get("result_count"),
            "error": result.get("error"), "elapsed_sec": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        return {"name": "search_provider_reachable", "ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(time.monotonic() - t0, 1)}


def _test_web_search_returns_real_results() -> dict:
    t0 = time.monotonic()
    try:
        result = web_search("Turkiye Cumhuriyet Merkez Bankasi politika faizi 2026", max_results=5)
        urls = [r.get("url", "") for r in result.get("results", [])]
        has_real_tcmb_hit = any("tcmb.gov.tr" in u for u in urls)
        return {
            "name": "web_search_returns_real_results", "ok": bool(urls) and has_real_tcmb_hit,
            "result_count": len(urls), "has_tcmb_gov_tr_hit": has_real_tcmb_hit, "sample_urls": urls[:3],
            "elapsed_sec": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        return {"name": "web_search_returns_real_results", "ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(time.monotonic() - t0, 1)}


def _test_web_url_extract() -> dict:
    t0 = time.monotonic()
    try:
        result = web_url_extract(_KNOWN_URL)
        text = result.get("text", "") or ""
        # A real fetch of this page should contain "2026" (its own PPK
        # calendar year) - an empty/near-empty extraction (the JS-SPA
        # problem found live against several TCMB/gov pages this
        # session) is a real, distinct failure mode from "can't reach it
        # at all".
        ok = len(text) > 20 and "2026" in text
        return {
            "name": "web_url_extract_known_url", "ok": ok, "extracted_chars": len(text),
            "kind": result.get("kind"), "elapsed_sec": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        return {"name": "web_url_extract_known_url", "ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(time.monotonic() - t0, 1)}


# --- agent-level: does it call the right tool and get the SPECIFIC facts
# right that were found wrong live -------------------------------------

AGENT_QUESTIONS = [
    {
        "id": "kmh_definition_regression",
        "note": (
            "Real hallucination found live: the model said 'KMH = Kredi "
            "Karti Harcamasi' (credit card spending) - wrong. KMH is "
            "'Kredili Mevduat Hesabi' (an overdraft account). Regression "
            "check for the system-prompt fix telling the model to verify "
            "terminology via source metadata or web_search instead of "
            "its own general knowledge."
        ),
        "question": "Bankacilikta KMH ne anlama gelir? Emin degilsen web_search ile dogrula, tahmin etme.",
        "must_include": ["kredili mevduat"],
        "must_not_include": ["kredi kartı harcaması"],
        "expect_tool_used": None,  # either lakehouse metadata or web_search is an acceptable path
    },
    {
        "id": "tr7_province_list_regression",
        "note": (
            "Real hallucination found live: the model called TR7 "
            "'Guneydogu Anadolu' with '6 il' - wrong on both the region "
            "name and the province count (it's actually 8 provinces: "
            "Nevsehir, Nigde, Kirikkale, Kirsehir, Aksaray, Kayseri, "
            "Sivas, Yozgat - a Central Anatolia grouping, not "
            "Southeastern)."
        ),
        "question": "TR7 IBBS bolge kodu hangi illeri kapsar ve hangi cografi bolgeye aittir? Emin degilsen dogrula.",
        "must_include": [],
        "must_not_include": ["güneydoğu anadolu"],
        "expect_tool_used": None,
    },
    {
        "id": "lakehouse_vs_live_web_cross_check",
        "note": (
            "Genuine hallucination-catching test: gold_fact_long's own "
            "policy-rate series should agree with what TCMB's real "
            "website currently says - if the agent fabricates a web "
            "result instead of actually fetching one, this is the kind "
            "of question that would expose it (the two numbers would "
            "either mismatch or the 'web' claim would have no real URL "
            "behind it)."
        ),
        "question": (
            "gold_fact_long'daki TCMB politika faizi serisinin (TP.BISPOLFAIZ.TUR) en guncel "
            "degerini sorgula. Sonra TCMB'nin kendi web sitesinde bu faiz oranini gercekten "
            "arayarak (web_search) tutarli olup olmadigini kontrol et. Kullandigin gercek URL'yi belirt."
        ),
        "must_include": [],
        "must_not_include": [],
        "expect_tool_used": "web_search",
    },
]


def _check_answer(item: dict, answer: str, tools_called: list[str]) -> dict:
    lowered = answer.lower()
    must_include = item.get("must_include", [])
    must_not_include = item.get("must_not_include", [])
    include_ok = all(s in lowered for s in must_include)
    exclude_ok = all(s not in lowered for s in must_not_include)
    tool_ok = True
    expect_tool = item.get("expect_tool_used")
    if expect_tool:
        tool_ok = expect_tool in tools_called

    if not must_include and not must_not_include and not expect_tool:
        return {"verdict": "REVIEW", "detail": "No auto-checkable rule - read by hand."}
    verdict = "PASS" if (include_ok and exclude_ok and tool_ok) else "FAIL"
    return {"verdict": verdict, "detail": f"include_ok={include_ok}, exclude_ok={exclude_ok}, tool_ok={tool_ok}"}


def run_agent_questions() -> list[dict]:
    results = []
    for item in AGENT_QUESTIONS:
        logger.info(f"run_agent_questions: asking '{item['id']}'")
        t0 = time.monotonic()
        answer, trace = ask(item["question"], return_trace=True)
        elapsed = round(time.monotonic() - t0, 1)
        tools_called = [t["name"] for t in trace]
        check = _check_answer(item, answer, tools_called)
        results.append({
            "id": item["id"], "note": item["note"], "question": item["question"],
            "answer": answer, "elapsed_sec": elapsed, "tools_called": tools_called, "auto_check": check,
        })
        logger.info(f"  [{check['verdict']}] {item['id']} ({elapsed}s, tools={tools_called})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Web tool + hallucination-regression benchmark")
    parser.add_argument("--skip-agent", action="store_true", help="Run only the fast provider-level checks, no LLM calls")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("WEB TOOL BENCHMARK")
    logger.info("=" * 70)

    provider_checks = [
        _test_search_provider_reachable(),
        _test_web_search_returns_real_results(),
        _test_web_url_extract(),
    ]
    for c in provider_checks:
        status = "OK" if c["ok"] else "FAIL"
        logger.info(f"  [{status}] {c['name']} ({c['elapsed_sec']}s)")

    agent_results = [] if args.skip_agent else run_agent_questions()

    report = {"provider_checks": provider_checks, "agent_questions": agent_results}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("WEB TOOL BENCHMARK SONUCU")
    print("=" * 70)
    for c in provider_checks:
        print(f"  [{'OK' if c['ok'] else 'FAIL':4s}] {c['name']:35s} {c['elapsed_sec']:5.1f}s")
    if agent_results:
        print("-" * 70)
        for r in agent_results:
            print(f"  [{r['auto_check']['verdict']:6s}] {r['id']:35s} {r['elapsed_sec']:6.1f}s tools={r['tools_called']}")
    print("-" * 70)
    print(f"Full report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
