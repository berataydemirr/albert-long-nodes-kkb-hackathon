"""Settles one specific claim with a direct A/B: does stuffing the full
lakehouse_schema dump (~14.8k chars / ~3.7k tokens) into the context
actually cause the multi-minute "thinking" phase?

The claim under test is that the agent is slow because it reads the whole
schema at once. The competing explanation is that wall time is dominated
by the REASONING tokens the model generates, and that prompt size is
close to free (prefill is parallel; decode is serial).

Both arms send the identical question and ask for the identical short
answer; the ONLY difference is whether the full schema dump is present in
the messages. If the schema were the cause, arm B would be dramatically
slower than arm A. The reasoning_effort arms then measure the knob that
the competing explanation predicts IS the real lever.

Run: python -m benchmarks.schema_payload_ab_test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config.settings import MIA_CHAT_MODEL
from src.agent.mia_client import get_client
from src.agent.tools import lakehouse_schema
from src.common.logging_setup import get_logger

logger = get_logger("schema_payload_ab_test")

REPORT_PATH = Path(__file__).with_name("schema_payload_ab_test_report.json")

QUESTION = (
    "Hanehalkının kredi kullanımını açıklamak için kullanılabilecek BDDK ve TCMB "
    "göstergelerini konu, frekans, birim ve kaynak açısından sınıflandır."
)
# Both arms ask for the SAME bounded output, so output length cannot
# confound the comparison between them.
BOUNDED = " En fazla 5 madde yaz, her madde tek satır olsun."


def _run(name: str, messages: list[dict], reasoning_effort: str | None, timeout: float = 240.0) -> dict:
    client = get_client().with_options(timeout=timeout)
    kwargs: dict = {"model": MIA_CHAT_MODEL, "messages": messages, "temperature": 0.1}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a failure is a datapoint here
        elapsed = time.monotonic() - t0
        logger.error(f"{name}: FAILED after {elapsed:.1f}s ({type(exc).__name__}: {exc})")
        return {"arm": name, "ok": False, "seconds": round(elapsed, 1),
                "reasoning_effort": reasoning_effort, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.monotonic() - t0
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    result = {
        "arm": name, "ok": True, "seconds": round(elapsed, 1),
        "reasoning_effort": reasoning_effort,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning,
    }
    logger.info(f"{name}: {elapsed:.1f}s | in={usage.prompt_tokens} out={usage.completion_tokens} "
                f"(reasoning={reasoning})")
    return result


def main() -> None:
    argparse.ArgumentParser(description="A/B: does the full schema dump cause the slow thinking phase?").parse_args()

    schema_text = json.dumps(lakehouse_schema(), ensure_ascii=False)
    logger.info(f"main: full schema dump is {len(schema_text)} chars (~{len(schema_text) // 4} tokens)")

    without_schema = [{"role": "user", "content": QUESTION + BOUNDED}]
    with_schema = [
        {"role": "user", "content": QUESTION + BOUNDED},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "lakehouse_schema", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": schema_text},
    ]

    results = [
        # The A/B pair: identical question and answer bound, schema present or not.
        _run("A_no_schema_effort_low", without_schema, "low"),
        _run("B_full_schema_effort_low", with_schema, "low"),
        # The knob the competing explanation says is the real lever.
        _run("C_full_schema_effort_minimal", with_schema, "minimal"),
        _run("D_full_schema_effort_medium", with_schema, "medium"),
    ]

    REPORT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 92)
    print("A/B: IS THE SCHEMA DUMP THE CAUSE OF THE SLOW THINKING PHASE?")
    print("=" * 92)
    print(f"{'arm':32s} {'effort':9s} {'sec':>7s} {'in_tok':>8s} {'out_tok':>8s} {'reasoning':>10s}")
    print("-" * 92)
    for r in results:
        if not r.get("ok"):
            print(f"{r['arm']:32s} {str(r.get('reasoning_effort')):9s} {r['seconds']:7.1f}  FAILED: {r['error'][:40]}")
            continue
        print(f"{r['arm']:32s} {str(r.get('reasoning_effort')):9s} {r['seconds']:7.1f} "
              f"{r['prompt_tokens']:8d} {r['completion_tokens']:8d} {str(r['reasoning_tokens']):>10s}")
    print("=" * 92)

    ok = {r["arm"]: r for r in results if r.get("ok")}
    a, b = ok.get("A_no_schema_effort_low"), ok.get("B_full_schema_effort_low")
    if a and b:
        extra_tokens = b["prompt_tokens"] - a["prompt_tokens"]
        delta = b["seconds"] - a["seconds"]
        print(f"SCHEMA COST: +{extra_tokens} prompt tokens changed wall time by {delta:+.1f}s")
    c, d = ok.get("C_full_schema_effort_minimal"), ok.get("D_full_schema_effort_medium")
    if c and d:
        print(f"EFFORT COST: minimal={c['seconds']}s vs medium={d['seconds']}s "
              f"(reasoning tokens {c['reasoning_tokens']} vs {d['reasoning_tokens']})")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
