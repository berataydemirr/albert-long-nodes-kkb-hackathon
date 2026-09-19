"""Isolates WHAT actually costs wall time on the MIA endpoint, instead of
guessing.

WHY THIS EXISTS: a broad "find every indicator" question was timing out,
and the obvious-sounding explanation was "we send the model too much data
(payload/schema too large)". That is a testable claim, and the two
candidate causes have opposite fixes:

  - if INPUT size (prompt tokens) drives latency, the fix is to send less
    context (RAG over the schema, fewer tool schemas, tighter catalog);
  - if OUTPUT size (completion tokens, and specifically REASONING tokens)
    drives latency, sending less context barely helps - the fix is the
    reasoning_effort knob, fewer rounds, and asking for shorter answers.

Prefill (reading input) is parallel on a GPU; decode (writing output) is
serial, one token at a time. So they are not remotely the same cost per
token, and the whole "is our payload too big?" question is answerable
with four controlled calls rather than an opinion.

Each case below varies ONE thing and reports tokens + seconds, so the
per-token cost of input and of output can be read off directly.

Run: python -m benchmarks.mia_latency_profile
     python -m benchmarks.mia_latency_profile --repeat 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config.settings import MIA_CHAT_MODEL
from src.agent.mia_client import get_client
from src.agent.orchestrator import _system_prompt
from src.agent.tools import TOOL_SCHEMAS
from src.common.logging_setup import get_logger

logger = get_logger("mia_latency_profile")

REPORT_PATH = Path(__file__).with_name("mia_latency_profile_report.json")

SHORT_ANSWER = "Sadece 'OK' yaz, başka hiçbir şey yazma."
# Deliberately forces a long chain of thought while sending almost no
# input - the mirror image of the big-prompt/short-answer case.
THINKING_QUESTION = (
    "Bir bankanın 12 aylık kümülatif net kârından aylık net kârı türetirken "
    "yıl sınırında hangi hatalar yapılır? Adım adım, ayrıntılı düşün ve uzun açıkla."
)


def _case(name: str, messages: list[dict], tools: list[dict] | None,
          reasoning_effort: str | None, timeout: float = 240.0) -> dict:
    client = get_client().with_options(timeout=timeout)
    kwargs: dict = {"model": MIA_CHAT_MODEL, "messages": messages, "temperature": 0.1}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a failed case is a datapoint, not a crash
        elapsed = time.monotonic() - t0
        logger.error(f"{name}: FAILED after {elapsed:.1f}s ({type(exc).__name__}: {exc})")
        return {"case": name, "ok": False, "seconds": round(elapsed, 1),
                "error": f"{type(exc).__name__}: {exc}", "reasoning_effort": reasoning_effort}

    elapsed = time.monotonic() - t0
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    result = {
        "case": name,
        "ok": True,
        "seconds": round(elapsed, 1),
        "reasoning_effort": reasoning_effort,
        "tool_schemas_sent": len(tools) if tools else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        # The two numbers the whole question turns on.
        "ms_per_prompt_token": round(elapsed * 1000 / prompt_tokens, 3) if prompt_tokens else None,
        "ms_per_completion_token": round(elapsed * 1000 / completion_tokens, 1) if completion_tokens else None,
    }
    logger.info(
        f"{name}: {elapsed:.1f}s | in={prompt_tokens} out={completion_tokens} "
        f"(reasoning={reasoning_tokens}) | {result['ms_per_completion_token']}ms per output token"
    )
    return result


def run(repeat: int = 1) -> list[dict]:
    system_prompt = _system_prompt()
    logger.info(f"run: system prompt is {len(system_prompt)} chars (~{len(system_prompt) // 4} tokens), "
                f"{len(TOOL_SCHEMAS)} tool schemas available")

    results: list[dict] = []
    for attempt in range(1, repeat + 1):
        logger.info(f"=== pass {attempt}/{repeat} ===")
        # 1. Floor: almost no input, almost no output.
        results.append(_case(
            "1_tiny_in_tiny_out", [{"role": "user", "content": SHORT_ANSWER}], None, None))

        # 2. BIG input, tiny output. If input size were the driver, this is
        #    where it would show up - same trivial answer, ~7000x the context.
        results.append(_case(
            "2_big_prompt_tiny_out",
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": SHORT_ANSWER}],
            None, None))

        # 3. Same big input, plus every tool schema. Isolates what the tool
        #    definitions themselves cost.
        results.append(_case(
            "3_big_prompt_all_tools_tiny_out",
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": SHORT_ANSWER}],
            TOOL_SCHEMAS, None))

        # 4. Tiny input, LONG reasoning. The mirror image of case 2.
        results.append(_case(
            "4_tiny_in_long_reasoning",
            [{"role": "user", "content": THINKING_QUESTION}], None, "medium"))

        # 5-7. The reasoning_effort knob on an identical real-shaped request.
        for effort in ("minimal", "low", "medium"):
            results.append(_case(
                f"5_real_shape_effort_{effort}",
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": "Lakehouse'ta hangi BDDK ve TCMB göstergeleri var? "
                                             "Konu, frekans, birim ve kaynak açısından sınıflandır."}],
                TOOL_SCHEMAS, effort))
    return results


def _print_summary(results: list[dict]) -> None:
    print("=" * 108)
    print("MIA LATENCY PROFILE - what actually costs wall time")
    print("=" * 108)
    header = (f"{'case':34s} {'effort':8s} {'sec':>7s} {'in_tok':>8s} {'out_tok':>8s} "
              f"{'reason':>7s} {'ms/out_tok':>11s}")
    print(header)
    print("-" * 108)
    for r in results:
        if not r.get("ok"):
            print(f"{r['case']:34s} {str(r.get('reasoning_effort') or '-'):8s} {r['seconds']:7.1f} "
                  f"{'FAILED: ' + r['error'][:50]}")
            continue
        print(f"{r['case']:34s} {str(r.get('reasoning_effort') or '-'):8s} {r['seconds']:7.1f} "
              f"{r['prompt_tokens']:8d} {r['completion_tokens']:8d} "
              f"{str(r['reasoning_tokens'] or '-'):>7s} {str(r['ms_per_completion_token']):>11s}")
    print("=" * 108)

    ok = [r for r in results if r.get("ok")]
    big_in = next((r for r in ok if r["case"] == "2_big_prompt_tiny_out"), None)
    tiny = next((r for r in ok if r["case"] == "1_tiny_in_tiny_out"), None)
    long_out = next((r for r in ok if r["case"] == "4_tiny_in_long_reasoning"), None)
    if big_in and tiny:
        extra_in = big_in["prompt_tokens"] - tiny["prompt_tokens"]
        extra_sec = big_in["seconds"] - tiny["seconds"]
        print(f"INPUT cost:  +{extra_in} prompt tokens cost {extra_sec:+.1f}s "
              f"({extra_sec * 1000 / extra_in:+.3f} ms per input token)" if extra_in else "")
    if long_out and tiny:
        extra_out = long_out["completion_tokens"] - tiny["completion_tokens"]
        extra_sec = long_out["seconds"] - tiny["seconds"]
        print(f"OUTPUT cost: +{extra_out} completion tokens cost {extra_sec:+.1f}s "
              f"({extra_sec * 1000 / extra_out:+.1f} ms per output token)" if extra_out else "")
    print("=" * 108)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile what drives MIA latency: input size vs output/reasoning")
    parser.add_argument("--repeat", type=int, default=1, help="Run every case N times (default 1)")
    args = parser.parse_args()

    results = run(args.repeat)
    REPORT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(results)
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
