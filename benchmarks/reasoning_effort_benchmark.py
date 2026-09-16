"""Measure the speed/quality trade-off of the model's reasoning_effort.

WHY THIS EXISTS:
The chat model is a reasoning model, and reasoning is where its wall time
goes - one long answer measured 18,212 reasoning tokens against 19,568
completion tokens, 93%. That makes reasoning_effort the largest latency
control available, larger than anything on our side of the wire.

Speed alone is the wrong thing to optimise, though: a fast answer that
drops the source citation or invents a Seri_Kod is worse than a slow
correct one. So each level is scored on both, and the cheap level only
wins if it still cites its sources and calls the right tools.

Run: python -m benchmarks.reasoning_effort_benchmark
Output: console table + benchmarks/reasoning_effort_report.json

This hits the real API and costs tokens; it is not part of the pytest
suite.
"""

import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows console codepage, see agent.py
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import orchestrator
from src.agent.series_resolver import catalog_block
from src.common.logging_setup import get_logger

logger = get_logger("reasoning_effort_benchmark")

REPORT_PATH = Path(__file__).parent / "reasoning_effort_report.json"

# 'none' and 'minimal' are accepted by the API schema but this model
# rejects some of them; unsupported levels are reported, not crashed on.
LEVELS = ["none", "minimal", "low", "medium", "high"]

QUESTIONS = [
    {
        "id": "causality",
        "question": "Konut kredisi faiz orani ile konut kredisi hacmi arasinda nedensellik var mi?",
        "expect_tools": {"causality_analysis"},
    },
    {
        "id": "anomaly",
        "question": "Konut kredisi faiz orani serisindeki aykiri donemleri bul ve acikla.",
        "expect_tools": {"anomaly_analysis"},
    },
    {
        "id": "catalog",
        "question": "Lakehouse'da hangi seriler var? Kisaca listele.",
        "expect_tools": set(),  # the catalog is in the system prompt already
    },
]


def _valid_series_codes() -> set[str]:
    codes = {line.split("|")[0].strip() for line in catalog_block().splitlines() if "|" in line}
    logger.info(f"_valid_series_codes: {len(codes)} codes from the injected catalog")
    return codes


def _cites_a_source(answer: str, valid_codes: set[str]) -> bool:
    """A usable answer names the series or the period it used. Without this
    check a level could 'win' purely by answering with less substance."""
    if any(code in answer for code in valid_codes):
        return True
    return bool(re.search(r"\b20(2[1-9])-\d{2}\b", answer))


def run_level(level: str, valid_codes: set[str]) -> dict:
    logger.info(f"run_level: {level} starting")
    results = []
    unsupported = None

    for item in QUESTIONS:
        t0 = time.monotonic()
        try:
            # Passed explicitly: otherwise the router's per-route effort
            # would replace the level under test.
            answer, trace, timing = orchestrator.ask(
                item["question"], return_trace=True, return_timing=True, reasoning_effort=level
            )
            error = None
        except Exception as exc:
            message = str(exc)
            # A level this model does not accept fails the same way every
            # time; record it once and move on rather than retrying.
            if "reasoning effort" in message.lower() or "literal_error" in message:
                logger.warning(f"run_level: {level} unsupported by the model ({message[:120]})")
                return {"level": level, "supported": False, "reason": message[:200]}
            logger.exception(f"run_level: {level} / {item['id']} raised")
            answer, trace, timing, error = "", [], None, message

        elapsed = time.monotonic() - t0
        called = {t["name"] for t in trace}
        expected = item["expect_tools"]
        tool_ok = (called == set()) if not expected else bool(expected & called)

        row = {
            "id": item["id"],
            "elapsed_sec": round(elapsed, 2),
            "llm_calls": timing["summary"]["llm_calls"] if timing else None,
            "tool_calls": timing["summary"]["tool_calls"] if timing else None,
            "called_tools": sorted(called),
            "tool_selection_ok": tool_ok,
            "answer_chars": len(answer),
            "cites_source": _cites_a_source(answer, valid_codes),
            "error": error,
            "answer": answer[:400],
        }
        logger.info(f"run_level: {level}/{item['id']} {row['elapsed_sec']}s tools={sorted(called)}")
        results.append(row)

    n = len(results)
    summary = {
        "level": level,
        "supported": True,
        "avg_sec": round(sum(r["elapsed_sec"] for r in results) / n, 2),
        "total_sec": round(sum(r["elapsed_sec"] for r in results), 2),
        "avg_llm_calls": round(sum((r["llm_calls"] or 0) for r in results) / n, 1),
        "tool_selection_accuracy": round(sum(r["tool_selection_ok"] for r in results) / n, 2),
        "citation_rate": round(sum(r["cites_source"] for r in results) / n, 2),
        "avg_answer_chars": round(sum(r["answer_chars"] for r in results) / n),
        "errors": sum(1 for r in results if r["error"]),
        "questions": results,
    }
    logger.info(f"run_level: {level} done avg={summary['avg_sec']}s citation={summary['citation_rate']}")
    return summary


def main() -> None:
    valid_codes = _valid_series_codes()
    if not valid_codes:
        print("No series catalog available. Run `python -m src.gold.build_gold` first.")
        raise SystemExit(1)

    levels = []
    for level in LEVELS:
        print(f"running level: {level} ...", flush=True)
        levels.append(run_level(level, valid_codes))

    REPORT_PATH.write_text(json.dumps(levels, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("REASONING EFFORT BENCHMARK")
    print("=" * 78)
    print(f"{'level':10s} {'avg':>8s} {'llm':>6s} {'tool sel':>9s} {'cite':>6s} {'chars':>7s} {'err':>4s}")
    print("-" * 78)
    for row in levels:
        if not row.get("supported"):
            print(f"{row['level']:10s} {'not supported by this model':>50s}")
            continue
        print(f"{row['level']:10s} {row['avg_sec']:7.2f}s {row['avg_llm_calls']:6.1f} "
              f"{row['tool_selection_accuracy']:9.2f} {row['citation_rate']:6.2f} "
              f"{row['avg_answer_chars']:7d} {row['errors']:4d}")
    print("-" * 78)
    print("A faster level only wins if tool selection and citation hold up.")
    print(f"Full report: {REPORT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
